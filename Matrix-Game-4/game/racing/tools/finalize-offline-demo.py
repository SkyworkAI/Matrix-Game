#!/usr/bin/env python3
"""Validate an aligned take, export reference.mp4, and compose UI/audio for review."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave


def probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate,pix_fmt:frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def valid_game_interval(previous, row, count, fps, physics_hz):
    dt = row["simulation_s"] - previous["simulation_s"]
    if math.isclose(dt, 1 / fps, abs_tol=1e-5):
        return True
    # A race can finish within the final frame; later substeps freeze the world.
    race = row.get("race", {})
    substeps = len(row.get("input_applied", []))
    if (race.get("mode") == "pursuit" and race.get("phase") == "results"
            and race.get("finished") is True and row.get("pursuit", {}).get("captured") is True):
        if previous.get("race", {}).get("phase") == "results":
            return substeps == 0 and math.isclose(dt, 0, abs_tol=1e-9)
        return (previous.get("race", {}).get("phase") == "racing"
                and 0 < substeps <= physics_hz / fps
                and math.isclose(dt, substeps / physics_hz, abs_tol=1e-5))
    return (row["frame"] == count - 1
            and previous.get("race", {}).get("phase") == "racing"
            and race.get("phase") == "results" and race.get("finished") is True
            and 0 < substeps < physics_hz / fps
            and math.isclose(dt, substeps / physics_hz, abs_tol=1e-5))


def validate(take):
    meta = json.loads((take / "session.json").read_text())
    if meta["status"] != "complete" or meta["frames"] < 1:
        raise ValueError("Take is incomplete or empty; retain the originals for diagnosis")
    count, fps = meta["frames"], meta["fps"]
    if meta["submitted_frames"] != count or meta["dropped_frames"] != 0:
        raise ValueError("Recording lost frames")
    for name in ("reference.mkv", "ui.mkv"):
        data = probe(take / name)
        stream = data["streams"][0]
        if (stream["width"], stream["height"], int(stream["nb_read_frames"])) != (meta["width"], meta["height"], count):
            raise ValueError(f"Wrong dimensions or frame count: {name}")
        if name == "ui.mkv" and stream["pix_fmt"] not in ("bgra", "rgba", "gbrap"):
            raise ValueError("UI alpha channel was lost")
        if len(data["frames"]) != count:
            raise ValueError(f"Missing timestamps: {name}")
        for index, frame in enumerate(data["frames"]):
            if not math.isclose(float(frame["best_effort_timestamp_time"]), index / fps, abs_tol=.0011):
                raise ValueError(f"Video timestamp drift: {name}, frame {index}")
    with wave.open(str(take / "audio.wav")) as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()) != (
            48000, 2, 2, count * 48000 // fps
        ):
            raise ValueError("Audio format/duration mismatch")
    previous = None
    rows = 0
    with (take / "frames.jsonl").open() as states:
        for rows, line in enumerate(states, 1):
            row = json.loads(line)
            index = rows - 1
            if row["frame"] != index or not math.isclose(row["pts_s"], index / fps, abs_tol=1e-9):
                raise ValueError("Noncontiguous frame metadata")
            if (row["audio_sample_start"], row["audio_sample_end"]) != (index * 48000 // fps, (index + 1) * 48000 // fps):
                raise ValueError("Audio intervals do not match frames")
            if previous is not None and "physics_hz" in meta:
                if row["physics_tick"] - previous["physics_tick"] != meta["physics_hz"] / fps:
                    raise ValueError("Unequal physics intervals")
                if not valid_game_interval(previous, row, count, fps, meta["physics_hz"]):
                    raise ValueError("Unequal game-time intervals")
            previous = row
    if rows != count:
        raise ValueError("Wrong metadata row count")
    applied = []
    with (take / "inputs.jsonl").open() as inputs:
        for line in inputs:
            event = json.loads(line)
            if not 0 <= event["next_frame"] <= count:
                raise ValueError("Input refers to an invalid frame")
            if event["event"] == "applied":
                applied.append(event["next_frame"])
    if "physics_hz" in meta and applied != list(range(count)):
        raise ValueError("Missing per-frame applied inputs")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--rendered", type=Path, help="Same-FPS, same-frame-count model output to composite instead")
    args = parser.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            parser.error(f"Missing {tool}")
    take = args.take.resolve()
    meta = validate(take)
    print(f"PASS: {meta['frames']} frames, {meta['fps']} FPS, {meta['duration_s']:.3f}s; RGB/UI/audio/input aligned", flush=True)
    if args.check_only:
        return
    if args.rendered:
        data = probe(args.rendered)
        stream = data["streams"][0]
        if (stream["width"], stream["height"], int(stream["nb_read_frames"])) != (meta["width"], meta["height"], meta["frames"]):
            parser.error("Model output must have identical dimensions and frame count; do not silently pad or drop frames")
        for index, frame in enumerate(data["frames"]):
            if not math.isclose(float(frame["best_effort_timestamp_time"]), index / meta["fps"], abs_tol=.0011):
                parser.error("Model output timestamps/FPS do not match this take")
    else:
        subprocess.run(["ffmpeg", "-n", "-v", "warning", "-i", str(take / "reference.mkv"),
                        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(take / "reference.mp4")], check=True)
    output = take / ("demo.mp4" if args.rendered else "preview.mp4")
    subprocess.run([
        "ffmpeg", "-n", "-v", "warning", "-i", str(args.rendered or take / "reference.mkv"),
        "-i", str(take / "ui.mkv"), "-i", str(take / "audio.wav"),
        "-filter_complex", (
            f"[0:v]settb=AVTB,setpts=N/({meta['fps']}*TB),format=rgb24[base];"
            f"[1:v]settb=AVTB,setpts=N/({meta['fps']}*TB),format=rgba[ui];"
            "[base][ui]overlay=alpha=premultiplied:format=rgb[v]"),
        "-map", "[v]", "-map", "2:a:0", "-frames:v", str(meta["frames"]),
        "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
    ], check=True)
    print(output)


if __name__ == "__main__":
    main()
