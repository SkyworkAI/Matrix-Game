#!/usr/bin/env python3
"""Add the racing demo's input-display style to a recorded shooter demo."""
import argparse
from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import wave

from PIL import Image, ImageDraw, ImageFont


ALIASES = {"W": "up", "Up": "up", "S": "down", "Down": "down",
           "A": "left", "Left": "left", "D": "right", "Right": "right",
           "Space": "space", "Left Shift": "shift", "Right Shift": "shift",
           **{key: key for key in ("1", "2", "3", "R", "C", "F", "T", "H")}}
BUTTONS = {1: "LMB", 3: "RMB"}
PANEL_SIZE = (204, 136)
POSITION = (14, 264)


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


class InputTimeline:
    def __init__(self, events, fps, frame_count, lead_ms):
        if frame_count < 1 or fps <= 0 or not 0 <= lead_ms <= 1000:
            raise ValueError("Invalid input timeline settings")
        self.end = frame_count - 1
        self.lead_frames = Fraction(str(lead_ms)) * Fraction(str(fps)) / 1000
        self.frames, self.states, self.transitions = [], [], []
        self.pulses = defaultdict(set)
        held, pressed_at = set(), {}
        previous = -1
        for event in events:
            frame = event["next_frame"]
            if not isinstance(frame, int) or not 0 <= frame <= frame_count or frame < previous:
                raise ValueError("Invalid or out-of-order input frame")
            previous = frame
            kind = event["type"]
            if kind == "initial":
                if self.frames or frame != 0:
                    raise ValueError("Invalid initial input state")
                state = event["state"]
                held = {k for k in state["keys"] if k in ALIASES}
                held.update(k for k, flag in (("LMB", "fire"), ("RMB", "ads")) if state[flag])
            elif kind == "key" and event["key"] in ALIASES:
                if event.get("repeat"):
                    continue
                key = event["key"]
            elif kind == "button" and event["button"] in BUTTONS:
                key = BUTTONS[event["button"]]
            else:
                continue
            if kind != "initial":
                if not self.frames:
                    raise ValueError("Missing initial input state")
                canonical = ALIASES.get(key, key)
                if event["down"]:
                    if key not in held:
                        pressed_at[key] = frame
                    held.add(key)
                else:
                    held.discard(key)
                    # A press/release received within one video frame still gets
                    # one visible pulse (e.g. the game's latched fire/jump input).
                    if pressed_at.pop(key, None) == frame:
                        self.pulses[frame].add(canonical)
                self.transitions.append({"source_frame": frame,
                                         "display_frame": max(0, math.ceil(frame - self.lead_frames)),
                                         "key": canonical, "down": event["down"]})
            self.frames.append(frame)
            self.states.append(frozenset(ALIASES.get(k, k) for k in held))
        if not self.frames or self.frames[0] != 0:
            raise ValueError("Missing initial input state")

    def at(self, frame):
        source = min(Fraction(self.end), Fraction(frame) + self.lead_frames)
        index = bisect_right(self.frames, source) - 1
        state = self.states[index]
        # Do not stretch a last-frame tap over the entire clamped tail.
        pulse = self.pulses.get(math.floor(Fraction(frame) + self.lead_frames), set())
        return state | pulse


def load_font(size, bold=False):
    paths = [f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
             f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def keyboard_layer(held, lead_ms):
    image = Image.new("RGBA", PANEL_SIZE)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, 203, 135), radius=5, fill=(10, 15, 18, 192))
    draw.text((8, 5), f"INPUT  -{lead_ms:g} ms", font=load_font(10), fill=(202, 212, 218, 255))

    def key_box(key, box, label=None):
        active = key in held
        draw.rounded_rectangle(box, radius=3,
                               fill=(82, 213, 200, 255) if active else (36, 47, 53, 240),
                               outline=(143, 230, 222, 255) if active else (104, 119, 126, 255))
        color = (6, 33, 37, 255) if active else (230, 236, 238, 255)
        if label:
            x1, y1, x2, y2 = box
            draw.text(((x1 + x2) / 2, (y1 + y2) / 2), label, anchor="mm",
                      font=load_font(10, True), fill=color)
        return color

    for key, (x, y) in {"up": (46, 21), "left": (16, 51), "down": (46, 51), "right": (76, 51)}.items():
        color = key_box(key, (x, y, x + 27, y + 26))
        cx, cy = x + 13, y + 13
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[key]
        tip = (cx + dx * 6, cy + dy * 6)
        draw.line((cx - dx * 5, cy - dy * 5, *tip), fill=color, width=2)
        draw.line((cx - dy * 4, cy + dx * 4, *tip, cx + dy * 4, cy - dx * 4), fill=color, width=2)
    key_box("space", (16, 81, 103, 99), "SPACE")
    key_box("LMB", (112, 21, 149, 47), "LMB")
    key_box("RMB", (155, 21, 192, 47), "RMB")
    for i, key in enumerate(("1", "2", "3")):
        x = 112 + i * 28
        key_box(key, (x, 51, x + 24, 77), key)
    key_box("shift", (112, 81, 192, 99), "SHIFT")
    for i, key in enumerate(("R", "C", "F", "T", "H")):
        x = 16 + i * 36
        key_box(key, (x, 107, x + 27, 125), key)
    return image


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"]


def check_video(path, frames, fps):
    streams = probe(path)
    video = [s for s in streams if s["codec_type"] == "video"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise ValueError("Expected one video and one audio track")
    v, a = video[0], audio[0]
    if (v["width"], v["height"], int(v["nb_read_frames"]), Fraction(v["avg_frame_rate"])) != (864, 480, frames, fps):
        raise ValueError("Wrong video size/frame count/FPS")
    for stream in (v, a):
        if abs(float(stream["start_time"])) > .001 or abs(float(stream["duration"]) - frames / fps) > .001:
            raise ValueError("Audio/video duration or start mismatch")


def trim_start_frame(seconds, fps, frame_count):
    frame = Fraction(str(seconds)) * Fraction(str(fps))
    if frame.denominator != 1 or not 0 <= frame < frame_count:
        raise ValueError("Trim start must be frame-aligned and leave a nonempty video")
    return int(frame)


def composition_filter(fps, start_frame=0):
    return (f"[1:v]trim=start_frame={start_frame},settb=expr=1/{fps},setpts=N[base];"
            f"[0:v]settb=expr=1/{fps},setpts=N[keys];"
            f"[base][keys]overlay=x={POSITION[0]}:y={POSITION[1]}:alpha=straight:format=auto:shortest=1[v]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keyboard-lead-ms", type=float, default=150)
    parser.add_argument("--trim-start-seconds", type=float, default=0)
    args = parser.parse_args()
    meta = json.loads((args.take / "session.json").read_text())
    if meta["status"] != "complete" or meta["fps"] != 48 or meta["dropped_frames"]:
        raise ValueError("Expected a complete 48 FPS recording without dropped frames")
    count, fps = meta["frames"], meta["fps"]
    start = trim_start_frame(args.trim_start_seconds, fps, count)
    output_count = count - start
    timeline = InputTimeline(read_jsonl(args.take / "inputs.jsonl"), fps, count, args.keyboard_lead_ms)
    check_video(args.video, count, fps)
    if start:
        with wave.open(str(args.take / "audio.wav")) as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()) != (48000, 2, 2, count * 1000):
                raise ValueError("Original PCM audio does not match the 48 FPS recording")
    output = args.output.resolve()
    temporary = output.with_name(output.stem + ".partial.mp4")
    if output.exists() or temporary.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".ffmpeg.log").open("w") as log:
        command = ["ffmpeg", "-nostdin", "-n", "-v", "warning", "-f", "rawvideo", "-pix_fmt", "rgba",
                   "-s", f"{PANEL_SIZE[0]}x{PANEL_SIZE[1]}", "-r", str(fps), "-i", "pipe:0",
                   "-threads", "2", "-i", str(args.video)]
        if start:
            # Cut the original PCM on the same frame boundary, without AAC priming offsets.
            command += ["-i", str(args.take / "audio.wav")]
        command += ["-filter_complex_threads", "2", "-filter_complex", composition_filter(fps, start),
                    "-map", "[v]", "-t", str(output_count / fps), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "16", "-pix_fmt", "yuv420p", "-threads", "4"]
        if start:
            command += ["-map", "2:a:0", "-af", f"atrim=start_sample={start * 1000}:end_sample={count * 1000},asetpts=N/SR/TB",
                        "-c:a", "aac", "-b:a", "192k"]
        else:
            command += ["-map", "1:a:0", "-c:a", "copy"]
        command += ["-movflags", "+faststart", str(temporary)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        cache = {}
        try:
            for frame in range(start, count):
                held = timeline.at(frame)
                if held not in cache:
                    cache[held] = keyboard_layer(held, args.keyboard_lead_ms).tobytes()
                encoder.stdin.write(cache[held])
                if (frame + 1) % 960 == 0 or frame + 1 == count:
                    print(f"OVERLAY {frame + 1 - start}/{output_count}", flush=True)
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg failed; see the output .ffmpeg.log")
        finally:
            if encoder.poll() is None:
                encoder.terminate()
                try:
                    encoder.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
    check_video(temporary, output_count, fps)
    temporary.rename(output)
    report = {"status": "complete", "take": str(args.take.resolve()), "source": str(args.video.resolve()),
              "output": str(output), "frames": output_count, "source_frames": count,
              "trim_start_frame": start, "trim_start_s": start / fps, "fps": fps, "duration_s": output_count / fps,
              "keyboard_lead_ms": args.keyboard_lead_ms, "latency_is_simulated": True,
              "sampling": "held(t + lead), clamped to last frame; one-frame pulse for sub-frame taps",
              "quantization": "ceil(event_frame - lead_ms * fps / 1000); error < one video frame",
              "audio": "trimmed original PCM on video frame boundary, then AAC encoded" if start else "copied without re-encoding or time shift",
              "ui": "existing HUD preserved", "transition_frame_origin": "original recording before trim",
              "position": POSITION, "panel_size": PANEL_SIZE, "transitions": timeline.transitions,
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {output}", flush=True)


if __name__ == "__main__":
    main()
