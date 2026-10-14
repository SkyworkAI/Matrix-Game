#!/usr/bin/env python3
"""Check a castle take and export clean reference/preview or a rerendered demo."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import wave


def probe(path):
    data = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate:frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return json.loads(data.stdout)


def check_video(path, count, fps, size):
    data = probe(path)
    stream = data["streams"][0]
    if int(stream["nb_read_frames"]) != count or (stream["width"], stream["height"]) != size:
        raise ValueError(f"Video frame count/size mismatch: {path}")
    num, den = map(int, stream["avg_frame_rate"].split("/"))
    if not math.isclose(num / den, fps, abs_tol=1e-5):
        raise ValueError(f"Video FPS mismatch: {path}")
    pts = [float(f["best_effort_timestamp_time"]) for f in data["frames"]]
    if len(pts) != count or any(abs(t - n / fps) > .0011 for n, t in enumerate(pts)):
        raise ValueError(f"Video timestamps mismatch: {path}")


def validate(take):
    manifest = json.loads((take / "session.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Recording did not finish cleanly")
    count, fps = manifest["frames"], manifest["fps"]
    if count < 1 or fps != 48 or manifest["speed"] != 2:
        raise ValueError("Expected a nonempty 48 FPS / 2x take")
    for name in ("reference.mkv", "ui.mkv"):
        check_video(take / name, count, fps, (manifest["width"], manifest["height"]))
    with wave.open(str(take / "audio.wav")) as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()) != (48000, 2, 2, count * 1000):
            raise ValueError("Audio format/duration mismatch")
    rows = [json.loads(line) for line in (take / "frames.jsonl").read_text().splitlines()]
    if len(rows) != count:
        raise ValueError("State count mismatch")
    previous_step, previous_time = manifest["simulation_step_start"], manifest["simulation_start"]
    event_ids = set()
    for frame, row in enumerate(rows):
        steps = row["physics_step"] - previous_step
        terminal = frame == count - 1 and row["dead"]
        if steps != 5 and not (terminal and 1 <= steps <= 5):
            raise ValueError(f"Unexpected simulation interval at frame {frame}")
        if len(row["input_applied"]) != steps or row["physics_step_start"] != previous_step:
            raise ValueError(f"Input/physics mismatch at frame {frame}")
        if not math.isclose(row["simulation_s"] - previous_time, steps / 120, abs_tol=1e-6):
            raise ValueError(f"Game-time mismatch at frame {frame}")
        if row["frame"] != frame or abs(row["pts_s"] - frame / fps) > 1e-8:
            raise ValueError("State PTS mismatch")
        if (row["audio_sample_start"], row["audio_sample_end"]) != (frame * 1000, (frame + 1) * 1000):
            raise ValueError("Audio interval mismatch")
        for event in row["events"]:
            if event["id"] in event_ids or not previous_time <= event["simulation_s"] <= row["simulation_s"] + 1e-6:
                raise ValueError("Duplicated or misaligned event")
            event_ids.add(event["id"])
        previous_step, previous_time = row["physics_step"], row["simulation_s"]
    inputs = [json.loads(line) for line in (take / "inputs.jsonl").read_text().splitlines()]
    if not inputs or inputs[0]["type"] != "initial" or any(not 0 <= event["next_frame"] <= count for event in inputs):
        raise ValueError("Invalid raw input trace")
    return manifest


def encode(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-n", *args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--rendered", type=Path)
    args = parser.parse_args()
    manifest = validate(args.take)
    print(f"PASS: {manifest['frames']} frames, 48 FPS, {manifest['duration_s']:.3f} presentation seconds, 2x game time")
    if args.check_only:
        return
    take = args.take
    if args.rendered:
        check_video(args.rendered, manifest["frames"], 48, (864, 480))
        base, name = args.rendered, "demo.mp4"
        scene_filter = "[0:v]settb=expr=1/48,setpts=N,format=rgba[scene]"
    else:
        if not (take / "reference.mp4").exists():
            encode(["-i", str(take / "reference.mkv"), "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(take / "reference.mp4")])
        base, name = take / "reference.mkv", "preview.mp4"
        scene_filter = "[0:v]settb=expr=1/48,setpts=N,scale=864:486:flags=bicubic,crop=864:480,format=rgba[scene]"
    # Matroska rounds UI timestamps to milliseconds; align by validated frame
    # index so framesync cannot extend the MP4 by one repeated terminal frame.
    filters = scene_filter + ";[1:v]settb=expr=1/48,setpts=N,scale=864:486:flags=bicubic,crop=864:480,format=rgba[ui];[scene][ui]overlay=alpha=premultiplied:format=auto:shortest=1[out]"
    encode(["-i", str(base), "-i", str(take / "ui.mkv"), "-i", str(take / "audio.wav"),
            "-filter_complex", filters, "-map", "[out]", "-map", "2:a:0", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(take / name)])
    check_video(take / name, manifest["frames"], 48, (864, 480))
    print(take / name)


if __name__ == "__main__":
    main()
