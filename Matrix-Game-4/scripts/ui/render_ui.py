#!/usr/bin/env python3
"""Local web UI for offline City Racing renders on the DGX.

Pick a recorded take (captures/take-NNN), a preset, denoising steps, size, prompt and an optional reference image, press Render, and the UI
runs the same steps as scripts/wsl/render_take.bat: copy the take to the DGX, render it with scripts/dgx/run_racing.sh, copy rendered.mp4 and
render.json back to outputs/<name>/. Past renders and the take's input video can be played side by side in two players.

The server listens on 127.0.0.1 only and needs only the Python standard library, ssh, scp and ffmpeg on PATH (key-based ssh to the DGX).
Usage:   python scripts/ui/render_ui.py [--port 8765] [--no-browser]
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shlex
import subprocess
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO = Path(__file__).resolve().parents[2]
DGX = "kschmid@10.74.11.118"
DGX_REPO = "/localhome/kschmid/Matrix-Game/Matrix-Game-4"
PRESETS = ("realtime", "quality")
STEPS = {"1": "1000", "2": "1000,500", "4": "1000,750,500,250", "8": "1000,875,750,625,500,375,250,125"}
SIZES = ("864x480", "1280x704", "1728x960")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,60}$")
TAKE_PATTERN = re.compile(r"^take-\d+$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_PROMPT = ("Daytime city, cloudy sky. Center: player-controlled modern four-door sedan with a curved roof and a separate rear trunk, "
                   "seen from behind in the foreground. Around: urban streets and multi-story buildings.")
PRESET_PROMPTS = {
    "Default": DEFAULT_PROMPT,
    "Sunny taxi": ("Sunny daytime city, bright warm sunlight, long sharp shadows on the road. Center: player-controlled worn yellow NYC taxi "
                   "cab with a TAXI roof sign and a rusty trunk, seen from behind in the foreground. Around: urban streets, stone and concrete "
                   "multi-story buildings, parked cars."),
    "Snowy sedan": ("Heavy snowstorm in a city, everything covered in thick white snow: snow-covered road, snow-covered sidewalks, snow piled "
                     "on cars, ledges, awnings and rooftops, snowflakes falling through the air, grey overcast sky, slush and dark tire tracks "
                     "in the snow, cold white-blue light, no dry pavement anywhere. Center: player-controlled blue four-door sedan with a "
                     "curved roof and a separate rear trunk, seen from behind in the foreground, dusted with snow. Around: snowy urban streets "
                     "and multi-story buildings, pedestrians in winter coats."),
    "Night city": ("Nighttime city, dark sky, streetlights and neon signs reflecting on wet asphalt. Center: player-controlled modern "
                    "four-door sedan with a curved roof and a separate rear trunk, headlights and taillights lit, seen from behind in the "
                    "foreground. Around: illuminated storefronts and multi-story buildings."),
}
EVENT_LINE = re.compile(r"^\s*(\d+)\s*:\s*(\S.*)$")
FRAMES_PATTERN = re.compile(r"(\d+) frames \|")
FPS_PATTERN = re.compile(r"\| ([\d.]+) processing FPS")
DONE_PATTERN = re.compile(r"finished in (\d+) s")
SSH = ["ssh", "-o", "BatchMode=yes", DGX]
SCP = ["scp", "-o", "BatchMode=yes"]

JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()


def take_info(path: Path) -> dict:
    manifest = json.loads((path / "session.json").read_text(encoding="utf-8"))
    return dict(name=path.name, frames=manifest.get("frames", 0), seconds=manifest.get("duration_s", 0), complete=manifest.get("status") == "complete")


def list_state() -> dict:
    takes = []
    for path in sorted((REPO / "captures").glob("take-*")):
        if (path / "session.json").exists():
            takes.append(take_info(path))
    renders = []
    for path in sorted((REPO / "outputs").glob("*/rendered.mp4"), key=lambda item: item.stat().st_mtime, reverse=True):
        settings = path.parent / "ui_params.json"
        renders.append(dict(name=path.parent.name, url=f"/files/outputs/{path.parent.name}/rendered.mp4",
                            params=json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else None))
    return dict(takes=takes, renders=renders, presets=list(PRESETS), steps=list(STEPS), sizes=list(SIZES), prompt_presets=PRESET_PROMPTS)


def convert_input(take: str) -> str:
    source = REPO / "captures" / take / "reference.mkv"
    target = REPO / "captures" / take / "reference.mp4"
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")
    if not target.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an",
                        "-movflags", "+faststart", str(target)], check=True)
    return f"/files/captures/{take}/reference.mp4"


class Job:
    def __init__(self, job_id: str, kind: str, params: dict, total_frames: int):
        self.id, self.kind, self.params, self.total = job_id, kind, params, total_frames
        self.result: dict | None = None
        self.fps = 0.0
        self.seconds = 0
        self.lines: list[str] = []
        self.status = "running"
        self.frames = 0
        self.output = ""
        self.started = time.time()

    def log(self, text: str) -> None:
        self.lines.append(text)
        match = FRAMES_PATTERN.search(text)
        if match:
            self.frames = int(match.group(1))
        fps = FPS_PATTERN.search(text)
        if fps:
            self.fps = float(fps.group(1))
        done = DONE_PATTERN.search(text)
        if done:
            self.seconds = int(done.group(1))

    def run(self, command: list[str]) -> None:
        self.log("$ " + " ".join(shlex.quote(part) for part in command[:6]) + (" ..." if len(command) > 6 else ""))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            self.log(line.rstrip())
        if process.wait() != 0:
            raise RuntimeError(f"command failed with exit code {process.returncode}")


def free_name(base: str) -> str:
    remote = f"cd {DGX_REPO} && n={base}; i=2; while [ -e outputs/$n ]; do n={base}_$i; i=`expr $i + 1`; done; echo $n"
    result = subprocess.run(SSH + [remote], capture_output=True, text=True, check=True)
    return result.stdout.strip().splitlines()[-1]


def build_captions(base: str, events: list[tuple[int, str]], total_frames: int) -> list[dict]:
    """One caption per chunk (frame 0 alone, then consecutive groups of 4), each "base. event" for the latest event at or before that chunk."""
    events = sorted(events)
    ranges = [(0, 0)]
    start = 1
    while start < total_frames:
        end = min(start + 3, total_frames - 1)
        ranges.append((start, end))
        start = end + 1
    records = []
    current = ""
    index = 0
    for frame_start, frame_end in ranges:
        while index < len(events) and events[index][0] <= frame_start:
            current = events[index][1]
            index += 1
        prompt = f"{base}. {current}" if current else base
        records.append(dict(frame_start=frame_start, frame_end=frame_end, prompt=prompt))
    return records


def render(job: Job) -> None:
    params = job.params
    try:
        take, preset = params["take"], params["preset"]
        name = free_name(params["name"])
        job.output = name
        job.log(f"output folder on the DGX: outputs/{name}")
        job.run(SCP + ["-r", str(REPO / "captures" / take), f"{DGX}:{DGX_REPO}/captures/"])
        extra = ""
        if preset == "quality":
            steps = STEPS[params["steps"]]
            extra += f" --steps {steps} --first-steps {steps}"
        if params.get("seed"):
            extra += f" --seed {int(params['seed'])}"
        if params.get("events"):
            total = take_info(REPO / "captures" / take)["frames"]
            records = build_captions(params["prompt"] or DEFAULT_PROMPT, params["events"], total)
            local_captions = REPO / "captures" / take / f"captions_{name}.jsonl"
            local_captions.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            remote_captions = f"captures/{take}/captions_{name}.jsonl"
            job.run(SCP + [str(local_captions), f"{DGX}:{DGX_REPO}/{remote_captions}"])
            extra += f" --prompt-mode dynamic --captions {remote_captions}"
            job.log(f"event prompt: {len(records)} chunk captions written to {remote_captions}")
        elif params.get("prompt"):
            extra += " --prompt " + shlex.quote(params["prompt"])
        if params.get("sink_image"):
            remote_image = f"captures/sink_{name}{Path(params['sink_image']).suffix.lower()}"
            job.run(SCP + [params["sink_image"], f"{DGX}:{DGX_REPO}/{remote_image}"])
            extra += f" --sink-image {remote_image}"
        size_env = f"MATRIX_GAME_RENDER_SIZE={params['size']} " if params["size"] != SIZES[0] else ""
        remote = (f"cd {DGX_REPO} && nvidia-smi --query-gpu=memory.free --format=csv,noheader && {size_env}bash scripts/dgx/run_racing.sh "
                  f"--take captures/{take} --preset {preset} --output outputs/{name} --{extra}")
        job.run(SSH + [remote])
        destination = REPO / "outputs" / name
        destination.mkdir(parents=True, exist_ok=True)
        job.run(SCP + [f"{DGX}:{DGX_REPO}/outputs/{name}/rendered.mp4", f"{DGX}:{DGX_REPO}/outputs/{name}/render.json", str(destination)])
        job.result = dict(kind="render", frames=job.frames, processing_fps=job.fps, render_seconds=job.seconds,
                          note="processing FPS includes video decode and encode on the CPU, so it is lower than the model-only FPS from Measure latency")
        (destination / "ui_params.json").write_text(json.dumps({**params, "output": name, "seconds": round(time.time() - job.started)}, indent=2), encoding="utf-8")
        job.status = "done"
    except Exception as error:
        job.log(f"ERROR: {error}")
        job.status = "failed"


def bench(job: Job) -> None:
    params = job.params
    try:
        take, preset = params["take"], params["preset"]
        job.run(SCP + ["-r", str(REPO / "captures" / take), f"{DGX}:{DGX_REPO}/captures/"])
        steps = STEPS[params["steps"]]
        extra = f" --steps {steps} --first-steps {steps}" if preset == "quality" else ""
        size_env = f"MATRIX_GAME_RENDER_SIZE={params['size']} " if params["size"] != SIZES[0] else ""
        report = f"outputs/bench_ui_{job.id}.json"
        remote = (f"cd {DGX_REPO} && nvidia-smi --query-gpu=memory.free --format=csv,noheader && env {size_env}PYTHONPATH=. .venv/bin/python "
                  f"scripts/dgx/benchmark_stream.py --scene racing --resources resources --take captures/{take} --preset {preset}{extra} "
                  f"--chunks 80 --warmup 20 --source-fps 24 > {report} 2> {report}.err")
        job.log("loading the model and timing 80 chunks (about 1 to 2 minutes)")
        job.run(SSH + [remote])
        text = subprocess.run(SSH + [f"cd {DGX_REPO} && cat {report}"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
        data = json.loads(text[text.index('{\n  "config"'):])
        stages = {key[:-3]: round(value["p50"], 1) for key, value in data["stage_ms"].items()}
        latency = data["streaming_latency_ms"]
        job.result = dict(kind="bench", preset=preset, steps=len(data["config"]["steps"]), size=params["size"],
                          chunk_p50_ms=round(data["chunk_wall_ms"]["p50"], 1), chunk_p95_ms=round(data["chunk_wall_ms"]["p95"], 1),
                          first_chunk_ms=round(data["first_chunk_wall_ms"]), fps=round(data["throughput_fps"]["mean"], 1), stage_p50_ms=stages,
                          latency_p50_ms=round(latency["total_p50"]), latency_p95_ms=round(latency["total_p95"]), chunk_fill_ms=round(latency["chunk_fill_ms"]),
                          source_fps=latency["source_fps"], real_time=data["real_time_at_source_fps"], load_seconds=round(data["model_load_seconds"]))
        job.status = "done"
    except Exception as error:
        job.log(f"ERROR: {error}")
        job.status = "failed"


def start_job(body: dict, kind: str) -> dict:
    take, preset, size = body.get("take", ""), body.get("preset", ""), body.get("size", SIZES[0])
    name = body.get("name") or f"racing_{take.replace('-', '')}_ui"
    if not TAKE_PATTERN.match(take) or not (REPO / "captures" / take / "session.json").exists():
        raise ValueError("choose a finished take")
    if preset not in PRESETS or size not in SIZES or body.get("steps", "4") not in STEPS or not NAME_PATTERN.match(name):
        raise ValueError("invalid preset, steps, size or name")
    sink = body.get("sink_image", "").strip().strip('"')
    if sink and (not Path(sink).is_file() or Path(sink).suffix.lower() not in IMAGE_SUFFIXES):
        raise ValueError("the reference image must be an existing .jpg, .png or .webp file")
    seed = str(body.get("seed", "")).strip()
    if seed and not seed.isdigit():
        raise ValueError("seed must be a whole number")
    total_frames = take_info(REPO / "captures" / take)["frames"]
    events: list[tuple[int, str]] = []
    for line_number, line in enumerate(body.get("events", "").splitlines(), 1):
        if not line.strip():
            continue
        match = EVENT_LINE.match(line)
        if not match:
            raise ValueError(f"event line {line_number} must look like '120: turning left', got {line!r}")
        frame = int(match.group(1))
        if frame >= total_frames:
            raise ValueError(f"event line {line_number}: frame {frame} is past the take's {total_frames} frames")
        events.append((frame, match.group(2).strip()))
    params = dict(take=take, preset=preset, steps=body.get("steps", "4"), size=size, name=name, prompt=body.get("prompt", "").strip(),
                  events=events, sink_image=sink, seed=seed)
    with JOB_LOCK:
        if any(job.status == "running" for job in JOBS.values()):
            raise ValueError("a render is already running")
        job = Job(str(int(time.time() * 1000)), kind, params, take_info(REPO / "captures" / take)["frames"])
        JOBS[job.id] = job
    threading.Thread(target=render if kind == "render" else bench, args=(job,), daemon=True).start()
    return dict(id=job.id)


class Handler(BaseHTTPRequestHandler):
    def reply(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def reply_json(self, data: dict, status: int = 200) -> None:
        self.reply(status, json.dumps(data).encode("utf-8"), "application/json")

    def serve_file(self, relative: str) -> None:
        parts = Path(unquote(relative)).parts
        if not parts or parts[0] not in ("outputs", "captures"):
            return self.reply_json({"error": "not found"}, 404)
        root = (REPO / parts[0]).resolve()
        path = (REPO / Path(*parts)).resolve()
        if root not in path.parents or not path.is_file():
            return self.reply_json({"error": "not found"}, 404)
        size = path.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get("Range", "")
        match = re.match(r"bytes=(\d*)-(\d*)$", header)
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            else:
                start = max(0, size - int(match.group(2)))
        end = min(end, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if match else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if match:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/":
                self.reply(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self.reply_json(list_state())
            elif url.path == "/api/input":
                take = query.get("take", [""])[0]
                if not TAKE_PATTERN.match(take):
                    raise ValueError("invalid take")
                self.reply_json(dict(url=convert_input(take)))
            elif url.path == "/api/job":
                job = JOBS[query.get("id", [""])[0]]
                since = int(query.get("since", ["0"])[0])
                self.reply_json(dict(status=job.status, lines=job.lines[since:], next=len(job.lines), frames=job.frames, total=job.total,
                                     output=job.output, kind=job.kind, result=job.result,
                                     url=f"/files/outputs/{job.output}/rendered.mp4" if job.status == "done" and job.kind == "render" else ""))
            elif url.path.startswith("/files/"):
                self.serve_file(url.path[len("/files/"):])
            else:
                self.reply_json({"error": "not found"}, 404)
        except (KeyError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
            self.reply_json({"error": str(error)}, 400)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/render", "/api/bench"):
            return self.reply_json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            self.reply_json(start_job(body, "render" if path == "/api/render" else "bench"))
        except (ValueError, KeyError) as error:
            self.reply_json({"error": str(error)}, 400)

    def log_message(self, format: str, *args) -> None:
        return


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Racing Render UI</title>
<style>
:root{--bg:#f4f3ef;--panel:#fff;--ink:#1d1b18;--muted:#6a645a;--line:#d9d4c8;--accent:#b4501f;--ok:#2f7a5a;--bad:#b3261e;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#141310;--panel:#1c1a16;--ink:#eee8dc;--muted:#a1998a;--line:#37332b;--accent:#e58a55;--ok:#6cc79f;--bad:#f2867e;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
.wrap{display:grid;grid-template-columns:340px 1fr;gap:16px;padding:16px;max-width:1500px;margin:0 auto}
@media (max-width:900px){.wrap{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;min-width:0}
h1{font-size:18px;margin:0 0 12px}h2{font-size:12px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin:16px 0 6px}
label{display:block;margin:10px 0 3px;color:var(--muted);font-size:12px}
select,input,textarea,button{width:100%;font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:7px 8px}
textarea{min-height:110px;resize:vertical}
button{cursor:pointer;background:var(--accent);color:#fff;border:0;font-weight:600;margin-top:14px;padding:10px}
button:disabled{opacity:.5;cursor:default}button.small{width:auto;padding:3px 9px;margin:0 4px 0 0;font-size:12px;background:var(--bg);color:var(--ink);border:1px solid var(--line)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.players{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media (max-width:700px){.players{grid-template-columns:1fr}}
video{width:100%;background:#000;border-radius:6px;aspect-ratio:864/480}.cap{color:var(--muted);font-size:12px;margin:4px 0}
progress{width:100%;height:10px}pre{margin:8px 0 0;max-height:220px;overflow:auto;font:12px/1.4 ui-monospace,Consolas,monospace;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;white-space:pre-wrap}
table{border-collapse:collapse;width:100%;font-size:12px}td{padding:3px 6px;border-bottom:1px solid var(--line)}td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.hist{display:grid;gap:6px;max-height:340px;overflow:auto}.item{display:flex;align-items:center;gap:6px;padding:6px 0;border-bottom:1px solid var(--line);flex-wrap:wrap}
.item span{flex:1;min-width:140px;overflow-wrap:anywhere}.status{font-weight:600}.status.done{color:var(--ok)}.status.failed{color:var(--bad)}
</style></head><body><div class="wrap">
<div class="panel"><h1>Racing render</h1>
<label for="take">Take</label><select id="take"></select>
<div class="row"><div><label for="preset">Preset</label><select id="preset"></select></div><div><label for="steps">Denoising steps</label><select id="steps"></select></div></div>
<div class="row"><div><label for="size">Size (unvalidated above 864x480)</label><select id="size"></select></div><div><label for="seed">Seed</label><input id="seed" placeholder="0"></div></div>
<label for="prompt">Prompt (empty = the scene's default caption; the base description when Events below are used)</label>
<div id="presets" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px"></div>
<textarea id="prompt" placeholder="Sunny daytime city, bright warm sunlight. Center: player-controlled yellow taxi seen from behind..."></textarea>
<label for="events">Events (optional): one "frame: what happens" per line, holds the base look and changes it per event</label>
<textarea id="events" placeholder="0: driving straight&#10;120: turning left at an intersection&#10;300: hits a snowbank, snow spraying"></textarea>
<label for="sink">Reference image, full path (empty = built-in grey sedan)</label><input id="sink" placeholder="C:\Users\kschmid\Downloads\image.jpg">
<label for="name">Output name (a number is added if it exists on the DGX)</label><input id="name" placeholder="racing_take001_ui">
<button id="go">Render on the DGX</button><button id="bench" style="background:var(--panel);color:var(--ink);border:1px solid var(--line)">Measure latency (take, preset, steps, size above)</button>
<h2>Progress</h2><progress id="bar" value="0" max="1"></progress><div class="cap" id="state">idle</div><pre id="log"></pre><h2>Metrics</h2><div id="metrics" class="cap">Render or measure to see numbers.</div></div>
<div><div class="panel"><div class="players">
<div><video id="va" controls loop muted></video><div class="cap" id="ca">A: nothing loaded</div></div>
<div><video id="vb" controls loop muted></video><div class="cap" id="cb">B: nothing loaded</div></div></div>
<div style="display:flex;gap:8px;margin-top:10px"><button class="small" id="play">Play both</button><button class="small" id="pause">Pause both</button><button class="small" id="inputA">Load take input into A</button></div></div>
<div class="panel" style="margin-top:16px"><h2 style="margin-top:0">Past renders</h2><div class="hist" id="hist"></div></div></div></div>
<script>
const $=id=>document.getElementById(id);let polling=null;
function fill(sel,values){sel.innerHTML=values.map(v=>`<option>${v}</option>`).join('')}
function load(slot,url,label){const v=$('v'+slot);v.src=url;v.load();$('c'+slot).textContent=slot.toUpperCase()+': '+label}
async function refresh(){const s=await (await fetch('/api/state')).json();
 fill($('take'),s.takes.map(t=>t.name));fill($('preset'),s.presets);fill($('steps'),s.steps);fill($('size'),s.sizes);$('steps').value='4';$('preset').value='quality';
 $('presets').innerHTML='';for(const [label,text] of Object.entries(s.prompt_presets)){const b=document.createElement('button');b.type='button';b.className='small';b.textContent=label;b.onclick=()=>{$('prompt').value=text};$('presets').appendChild(b)}
 $('hist').innerHTML='';for(const r of s.renders){const d=document.createElement('div');d.className='item';const p=r.params;
  const info=p?` (${p.preset}${p.preset==='quality'?', '+p.steps+' steps':''}, ${p.size})`:'';
  d.innerHTML=`<span>${r.name}${info}</span>`;
  for(const slot of ['a','b']){const b=document.createElement('button');b.className='small';b.textContent=slot.toUpperCase();b.onclick=()=>load(slot,r.url,r.name);d.appendChild(b)}
  $('hist').appendChild(d)}}
async function poll(id,since){const j=await (await fetch(`/api/job?id=${id}&since=${since}`)).json();
 const log=$('log');if(j.lines.length){log.textContent+=j.lines.join('\n')+'\n';log.scrollTop=log.scrollHeight}
 $('bar').max=j.total||1;$('bar').value=j.frames;$('state').innerHTML=`<span class="status ${j.status}">${j.status}</span> ${j.frames}/${j.total} frames ${j.output?'→ outputs/'+j.output:''}`;
 if(j.status==='running'){polling=setTimeout(()=>poll(id,j.next),1000)}else{$('go').disabled=false;$('bench').disabled=false;if(j.status==='done'){if(j.kind==='render'){load('b',j.url,j.output);refresh()}showMetrics(j.result)}}}
function showMetrics(r){if(!r)return;let rows;if(r.kind==='bench'){rows=[['preset / steps / size',r.preset+' / '+(r.preset==='quality'?r.steps:1)+' / '+r.size],['chunk time p50 / p95',r.chunk_p50_ms+' / '+r.chunk_p95_ms+' ms'],['first chunk',r.first_chunk_ms+' ms'],['FPS (model only)',r.fps]];for(const [k,v] of Object.entries(r.stage_p50_ms))rows.push(['stage '+k+' p50',v+' ms']);rows.push(['chunk fill at '+r.source_fps+' fps',r.chunk_fill_ms+' ms'],['latency p50 / p95 (fill + processing)',r.latency_p50_ms+' / '+r.latency_p95_ms+' ms'],['real time at '+r.source_fps+' fps',r.real_time?'yes':'no'],['model load',r.load_seconds+' s'])}else{rows=[['frames',r.frames],['processing FPS (includes video I/O)',r.processing_fps],['render time',r.render_seconds+' s']]}
 $('metrics').innerHTML='<table>'+rows.map(x=>'<tr><td>'+x[0]+'</td><td>'+x[1]+'</td></tr>').join('')+'</table>'+(r.note?'<div class="cap">'+r.note+'</div>':'')}
$('bench').onclick=async()=>{$('log').textContent='';$('go').disabled=true;$('bench').disabled=true;const body={take:$('take').value,preset:$('preset').value,steps:$('steps').value,size:$('size').value};const j=await (await fetch('/api/bench',{method:'POST',body:JSON.stringify(body)})).json();if(j.error){$('state').textContent=j.error;$('go').disabled=false;$('bench').disabled=false;return}poll(j.id,0)};
$('go').onclick=async()=>{$('log').textContent='';$('go').disabled=true;$('bench').disabled=true;
 const body={take:$('take').value,preset:$('preset').value,steps:$('steps').value,size:$('size').value,seed:$('seed').value,prompt:$('prompt').value,events:$('events').value,sink_image:$('sink').value,name:$('name').value};
 const r=await fetch('/api/render',{method:'POST',body:JSON.stringify(body)});const j=await r.json();
 if(j.error){$('state').textContent=j.error;$('go').disabled=false;$('bench').disabled=false;return}poll(j.id,0)};
$('inputA').onclick=async()=>{const j=await (await fetch('/api/input?take='+$('take').value)).json();if(j.error){$('state').textContent=j.error;return}load('a',j.url,'input of '+$('take').value)};
$('play').onclick=()=>{$('va').currentTime=0;$('vb').currentTime=0;$('va').play();$('vb').play()};$('pause').onclick=()=>{$('va').pause();$('vb').pause()};
refresh();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Racing render UI: {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
