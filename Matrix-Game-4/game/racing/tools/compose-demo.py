#!/usr/bin/env python3
"""Re-layout recorded HUD panels and add a time-shifted input visualization."""
import argparse
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ALIASES = {"W": "up", "Up": "up", "S": "down", "Down": "down",
           "A": "left", "Left": "left", "D": "right", "Right": "right",
           "Space": "space"}


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


class InputTimeline:
    def __init__(self, events, fps, frame_count, lead_ms):
        self.fps = Fraction(str(fps))
        self.lead = Fraction(str(lead_ms)) / 1000
        self.end = frame_count - 1
        self.frames, self.states = [], []
        held = set()
        self.transitions = []
        for event in events:
            frame = event["next_frame"]
            kind = event["event"]
            if kind == "initial_state":
                held = set(event["held_keys"])
            elif kind == "key" and event["key"] in ALIASES:
                if event["down"]:
                    held.add(event["key"])
                else:
                    held.discard(event["key"])
                first = max(0, math.ceil(Fraction(frame) - self.lead * self.fps))
                self.transitions.append({"source_frame": frame, "display_frame": first,
                                         "key": event["key"], "down": event["down"]})
            else:
                continue
            if self.frames and frame < self.frames[-1]:
                raise ValueError("Out-of-order input events")
            self.frames.append(frame)
            self.states.append(frozenset(ALIASES[key] for key in held if key in ALIASES))
        if not self.frames or self.frames[0] != 0:
            raise ValueError("Missing initial keyboard state")

    def at(self, frame):
        # Sample t + lead, not wall time; recording may have waited on disk I/O.
        source = min(Fraction(self.end), Fraction(frame) + self.lead * self.fps)
        index = bisect_right(self.frames, source) - 1
        return self.states[index] if index >= 0 else frozenset()


@dataclass
class Panel:
    name: str
    box: tuple
    background: tuple
    size: tuple
    position: tuple


def panel_layout(meta, hud_scale=.72, map_size=180):
    rw, rh = meta["render_width"], meta["render_height"]
    width, height = meta["width"], meta["height"]
    hw, hh = meta.get("hud_layout_width", rw), meta.get("hud_layout_height", rh)
    scale = meta.get("hud_render_scale", 1)
    if ((rw, rh) not in ((1280, 720), (1920, 1080))
            or (hw, hh, width, height) != (1280, 720, 864, 480)
            or scale != rw / hw or scale != rh / hh):
        raise ValueError("HUD extraction requires a 1280x720 logical HUD, rendered at 720p or 1080p and resized/cropped to 864x480")
    factor, crop_y = width / hw, 3

    def box(x, y, w, h):
        return (math.floor(x * factor) - 1, math.floor(y * factor - crop_y) - 1,
                math.ceil((x + w) * factor) + 1, math.ceil((y + h) * factor - crop_y) + 1)

    panels = []
    definitions = [
        ("timer", (24, 24, 382, 136), (5, 8, 9, 217)),
        ("navigation", (466, 24, 348, 88), (5, 8, 9, 219)),
        ("notice", (24, 174, 460, 34), (5, 7, 8, 204)),
    ]
    if meta.get("mode") == "pursuit":
        definitions = [
            ("pursuit", (24, 24, 360, 130), (6, 8, 9, 224)),
            ("notice", (40, 168, 384, 18), (0, 0, 0, 0)),
        ]
    definitions += [
        ("speed", (16, 622, 202, 69), (6, 9, 9, 209)),
        ("map", (808, 248, 448, 448), (6, 9, 9, 235)),
    ]
    for name, rect, background in definitions:
        bounds = box(*rect)
        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        scale = .85 if name == "speed" else hud_scale
        size = (map_size, map_size) if name == "map" else (round(w * scale), round(h * scale))
        position = {"timer": (14, 14), "pursuit": (14, 14), "navigation": ((width - size[0]) // 2, 14),
                    "notice": (14, 88), "speed": (14, height - 14 - size[1]),
                    "map": (width - 14 - size[0], height - 14 - size[1])}[name]
        panels.append(Panel(name, bounds, background, size, position))
    return panels


def gate_support(state, size):
    mask = Image.new("L", size)
    gate = state.get("gate")
    if not gate:
        return np.asarray(mask, dtype=bool)
    draw = ImageDraw.Draw(mask)
    thickness = max(6, min(100, math.ceil(400 / max(gate["distance_m"], 1))))
    for x1, y1, x2, y2 in gate["segments"]:
        draw.line((x1 * size[0], y1 * size[1], x2 * size[0], y2 * size[1]), fill=255, width=thickness)
    return np.asarray(mask, dtype=bool)


def split_panel(pixels, background, support, gate_color):
    """Recover gates on transparent/flat panel pixels, without fabricating hidden pixels.

    UI is already premultiplied. In flat areas U = B + (1 - B.a) * G.
    Opaque glyphs/road lines destroy the underlying information and are not inverted.
    """
    clean = pixels.copy()
    gates = np.zeros_like(pixels)
    if not support.any():
        return clean, gates
    source = pixels.astype(np.float32)
    color = np.array(gate_color, np.float32)
    nominal = 242 if gate_color[0] < .5 else 230
    pure_error = np.max(np.abs(source[..., :3] - source[..., 3:4] * color), axis=-1)
    pure = support & (pure_error <= 1.7) & (source[..., 3] > 0) & (source[..., 3] <= nominal + 1)
    gates[pure] = pixels[pure]
    clean[pure] = 0

    bg = np.array(background, np.float32)
    delta = source[..., :3] - bg[:3]
    weight = np.sum(delta * color, axis=-1) / np.sum(color * color)
    error = np.max(np.abs(delta - weight[..., None] * color), axis=-1)
    remaining = 1 - bg[3] / 255
    alpha = weight / remaining
    mixed = (support & ~pure & (weight >= 1.5) & (error <= 1.25)
             & (np.abs(source[..., 3] - bg[3] - weight) <= 1.6)
             & (alpha <= nominal + 20))
    recovered = np.minimum(alpha[mixed], nominal)
    gates[mixed, :3] = np.rint(recovered[:, None] * color).astype(np.uint8)
    gates[mixed, 3] = np.rint(recovered).astype(np.uint8)
    clean[mixed] = bg.astype(np.uint8)
    return clean, gates


def resize_premultiplied(pixels, size):
    # RGBA.resize would premultiply a second time; resize each already-premultiplied channel.
    result = np.stack([np.asarray(Image.fromarray(pixels[..., c]).resize(size, Image.Resampling.LANCZOS))
                       for c in range(4)], axis=-1)
    result[..., :3] = np.minimum(result[..., :3], result[..., 3:4])
    return result


def over(destination, layer, position):
    x, y = position
    h, w = layer.shape[:2]
    target = destination[y:y + h, x:x + w]
    value = layer.astype(np.uint16) + (target.astype(np.uint16)
                                      * (255 - layer[..., 3:4].astype(np.uint16)) + 127) // 255
    target[:] = np.minimum(value, 255).astype(np.uint8)


def premultiply(image):
    data = np.array(image)
    data[..., :3] = ((data[..., :3].astype(np.uint16) * data[..., 3:4] + 127) // 255).astype(np.uint8)
    return data


def load_font(size, bold=False):
    candidates = [f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
                  f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def keyboard_layer(held, lead_ms):
    im = Image.new("RGBA", (120, 106))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((0, 0, 119, 105), radius=5, fill=(10, 15, 18, 192))
    draw.text((8, 5), f"INPUT  -{lead_ms:g} ms", font=load_font(10), fill=(202, 212, 218, 255))
    keys = {"up": (46, 21), "left": (16, 51), "down": (46, 51), "right": (76, 51)}
    for key, (x, y) in keys.items():
        active = key in held
        draw.rounded_rectangle((x, y, x + 27, y + 26), radius=3,
                               fill=(82, 213, 200, 255) if active else (36, 47, 53, 240),
                               outline=(143, 230, 222, 255) if active else (104, 119, 126, 255))
        cx, cy = x + 13, y + 13
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[key]
        color = (6, 33, 37, 255) if active else (230, 236, 238, 255)
        tip = (cx + dx * 6, cy + dy * 6)
        draw.line((cx - dx * 5, cy - dy * 5, *tip), fill=color, width=2)
        draw.line((cx - dy * 4, cy + dx * 4, *tip, cx + dy * 4, cy - dx * 4), fill=color, width=2)
    draw.rounded_rectangle((16, 81, 103, 99), radius=3,
                           fill=(82, 213, 200, 255) if "space" in held else (36, 47, 53, 240),
                           outline=(104, 119, 126, 255))
    draw.text((60, 90), "SPACE", anchor="mm", font=load_font(10, True),
              fill=(6, 33, 37, 255) if "space" in held else (230, 236, 238, 255))
    return premultiply(im)


def reshape_ui(ui, state, panels):
    race = state.get("race", {})
    # Pursuit's frozen capture tail still contains the HUD, not a full-screen menu.
    capture_tail = (race.get("mode") == "pursuit" and race.get("phase") == "results"
                    and state.get("pursuit", {}).get("captured") is True)
    if race.get("phase", "racing") != "racing" and not capture_tail:
        return ui.copy()
    result = ui.copy()
    support = gate_support(state, (ui.shape[1], ui.shape[0]))
    color = (.35, .95, .63) if race.get("next", 0) + 1 == race.get("total") else (1, .77, .16)
    crops = []
    for panel in panels:
        x1, y1, x2, y2 = panel.box
        clean, gates = split_panel(ui[y1:y2, x1:x2], panel.background, support[y1:y2, x1:x2], color)
        result[y1:y2, x1:x2] = gates
        crops.append((panel, clean))
    for panel, pixels in crops:
        over(result, resize_premultiplied(pixels, panel.size), panel.position)
    return result


def read_exact(stream, count):
    parts, size = [], 0
    while size < count:
        part = stream.read(count - size)
        if not part:
            raise ValueError(f"Truncated video: received {size}/{count} bytes")
        parts.append(part)
        size += len(part)
    return b"".join(parts)


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams",
                             "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"]


def check_video(path, meta, expected):
    stream = next(s for s in probe(path) if s["codec_type"] == "video")
    if (stream["width"], stream["height"], int(stream["nb_read_frames"]), Fraction(stream["avg_frame_rate"])) != (
            meta["width"], meta["height"], expected, Fraction(meta["fps"])):
        raise ValueError(f"Wrong resolution/frame count/FPS: {path}")


def composition_filter(fps):
    # AVTB rounds some 48 FPS timestamps down, making framesync select the
    # previous overlay frame. Both inputs must use the same exact frame clock.
    return (f"[1:v]settb=expr=1/{fps},setpts=N,format=rgb24[base];"
            f"[0:v]settb=expr=1/{fps},setpts=N,format=rgba[ui];"
            "[base][ui]overlay=alpha=premultiplied:format=rgb:shortest=1[v]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take", type=Path)
    parser.add_argument("--rendered", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keyboard-lead-ms", type=float, default=200)
    parser.add_argument("--hud-scale", type=float, default=.72)
    parser.add_argument("--map-size", type=int, default=180)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if not (0 < args.hud_scale <= 1 and 100 <= args.map_size <= 240 and 0 <= args.keyboard_lead_ms <= 1000):
        parser.error("Invalid HUD/input settings")
    meta = json.loads((args.take / "session.json").read_text())
    if meta["status"] != "complete" or meta["dropped_frames"] or meta["ui_alpha"] != "premultiplied":
        raise ValueError("Require a complete, lossless, premultiplied-alpha take")
    check_video(args.rendered, meta, meta["frames"])
    total = min(meta["frames"], args.max_frames or meta["frames"])
    if total < 1:
        parser.error("Empty selection")
    fps, w, h = meta["fps"], meta["width"], meta["height"]
    panels = panel_layout(meta, args.hud_scale, args.map_size)
    inputs = InputTimeline(read_jsonl(args.take / "inputs.jsonl"), fps, meta["frames"], args.keyboard_lead_ms)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    if output.exists() or temporary.exists():
        raise FileExistsError(output)
    logs = output.with_suffix(".ffmpeg.log")
    duration = total / fps
    with logs.open("w") as log:
        decoder = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", str(args.take / "ui.mkv"),
                                    "-frames:v", str(total), "-f", "rawvideo", "-pix_fmt", "rgba", "-threads", "2", "pipe:1"],
                                   stdout=subprocess.PIPE, stderr=log)
        encoder = subprocess.Popen([
            "ffmpeg", "-nostdin", "-n", "-v", "warning", "-f", "rawvideo", "-pixel_format", "rgba",
            "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", "pipe:0",
            "-threads", "2", "-i", str(args.rendered), "-i", str(args.take / "audio.wav"),
            "-filter_complex_threads", "2", "-filter_complex",
            composition_filter(fps),
            "-map", "[v]", "-map", "2:a:0", "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", "-threads", "4",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temporary)], stdin=subprocess.PIPE, stderr=log)
        cache = {}
        count = 0
        try:
            for frame, state in enumerate(read_jsonl(args.take / "frames.jsonl")):
                if frame >= total:
                    break
                if state["frame"] != frame or abs(state["pts_s"] - frame / fps) > 1e-8:
                    raise ValueError("Noncontiguous frame metadata")
                ui = np.frombuffer(read_exact(decoder.stdout, w * h * 4), np.uint8).reshape(h, w, 4)
                layer = reshape_ui(ui, state, panels)
                held = inputs.at(frame)
                if held not in cache:
                    cache[held] = keyboard_layer(held, args.keyboard_lead_ms)
                over(layer, cache[held], (14, h - 14 - 42 - 10 - 106))
                encoder.stdin.write(layer.tobytes())
                count += 1
                if count % 960 == 0 or count == total:
                    print(f"COMPOSE {output.name}: {count}/{total}", flush=True)
            if count != total or decoder.stdout.read(1):
                raise ValueError("Unexpected UI/metadata frame count")
            decoder.stdout.close()
            encoder.stdin.close()
            if decoder.wait() or encoder.wait():
                raise RuntimeError(f"ffmpeg failed; see {logs}")
        finally:
            for process in (decoder, encoder):
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    check_video(temporary, meta, total)
    streams = probe(temporary)
    audio = next(s for s in streams if s["codec_type"] == "audio")
    if abs(float(audio["duration"]) - duration) > .025 or float(audio.get("start_time", 0)) != 0:
        raise ValueError("Audio duration/start mismatch")
    temporary.rename(output)
    report = {
        "status": "complete", "take": str(args.take.resolve()), "rendered": str(args.rendered.resolve()),
        "output": str(output), "frames": total, "fps": fps, "duration_s": duration,
        "keyboard_lead_ms": args.keyboard_lead_ms, "keyboard_sampling": "held(t + lead), clamped to last source frame",
        "keyboard_quantization": "ceil(event_frame - lead_ms * fps / 1000); error < one video frame",
        "latency_is_simulated": True, "audio": "original timing, not shifted",
        "hud_scale": args.hud_scale,
        "speed_scale": .85,
        "map_size": args.map_size,
        "ui_layout": "compact_pursuit" if meta.get("mode") == "pursuit" else "compact_time_trial",
        "gate_handling": "Original pixels outside HUD; invert flat translucent panels in place; do not invent gates hidden by opaque HUD glyphs",
        "keyboard_transitions": inputs.transitions,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {output} ({total} frames, {duration:.4f}s)", flush=True)


if __name__ == "__main__":
    main()
