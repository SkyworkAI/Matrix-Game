#!/usr/bin/env python3
"""Measure chunk size, per-chunk latency and throughput of the Matrix-Game stream on a recorded take (or any low-poly video).

Runs the same model loader and stream the offline render uses, with the stream's CUDA-event profiling switched on, and prints one JSON report:
  chunk size           RGB frames per chunk (1 for the first chunk, 4 afterwards) and the latent shape one chunk produces
  chunk wall time      host wall clock of one `push` (encode + denoise + decode + clean KV), after the CUDA work finished
  stage times          encode / denoise / clean_kv / decode in milliseconds (CUDA events; they include host launch gaps)
  throughput           frames per second = 4 / mean chunk wall time, measured after the warmup chunks
  streaming latency    the time from the last frame of a chunk being available to its 4 rendered frames being ready, plus the time the
                       source needs to produce those 4 frames at --source-fps (a game at 24 fps takes 4/24 s = 167 ms to fill a chunk)
Usage:   python scripts/dgx/benchmark_stream.py --take captures/take-001 --preset realtime --chunks 100 [--source-fps 24] [model arguments]
Model arguments are those of `matrix-game render` (--scene, --resources, --preset, --steps, --first-steps, --vae-backend, --dit-runtime...).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image

from matrix_game.inference.loader import ModelComponents
from matrix_game.inference.offline import frame_blocks
from matrix_game.inference.preprocessing import crop_and_resize, frames_to_tensor
from matrix_game.inference.resolution import HEIGHT, WIDTH
from matrix_game.scenes import SCENES, add_model_arguments, resolve_model_arguments


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def summary(values):
    return dict(mean=sum(values) / len(values), p50=percentile(values, .5), p95=percentile(values, .95), maximum=max(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(ap)
    ap.set_defaults(preset="quality")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Clean low-poly RGB video")
    source.add_argument("--take", type=Path, help="Completed game recording directory")
    ap.add_argument("--chunks", type=int, default=100, help="chunks to run, including the first one-frame chunk")
    ap.add_argument("--warmup", type=int, default=10, help="chunks excluded from the statistics")
    ap.add_argument("--source-fps", type=float, default=24.0, help="frame rate of the live source, for the streaming latency estimate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.chunks <= args.warmup + 1:
        ap.error("--chunks must exceed --warmup by at least 2")
    if args.take:
        manifest = json.loads((args.take / "session.json").read_text())
        scene = manifest.get("scene", manifest.get("game"))
        if args.scene == "racing" and manifest.get("mode") in ("drive", "time_trial", "pursuit"):
            scene = "racing"
        if scene not in SCENES or manifest.get("prompt") != SCENES[scene]["prompt"]:
            ap.error("Recording scene/prompt does not match release profiles")
        args.scene, args.input = scene, args.take / "reference.mkv"
    resolve_model_arguments(args)
    if args.dit_runtime.startswith("rtx5090"):
        from matrix_game.inference.rtx5090 import FastStream as Stream
    else:
        from matrix_game.inference.stream import CastleStream as Stream

    torch.manual_seed(args.seed)
    load_started = time.perf_counter()
    components = ModelComponents(args)
    components.load()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    stream = Stream(components.gen, components.vae, components.prompt, seed=args.seed, vae_backend=args.vae_backend, steps=args.steps,
                    first_steps=args.first_steps, sink_latent=components.sink_latent, sink_mode=args.sink_mode,
                    sink_rope_mode=args.sink_rope_mode, profile=True)

    reader = imageio_ffmpeg.read_frames(str(args.input), pix_fmt="rgb24")
    info = next(reader)
    width, height = info["size"]
    rows = []
    try:
        for index, (raw, valid) in enumerate(frame_blocks(reader)):
            if index >= args.chunks or valid < len(raw):
                break
            frames = [crop_and_resize(Image.fromarray(np.frombuffer(r, np.uint8).reshape(height, width, 3)), HEIGHT, WIDTH) for r in raw]
            pixels = frames_to_tensor(frames)
            torch.cuda.synchronize()
            started = time.perf_counter()
            chunk = stream.push(pixels, frame_start=sum(row["frames"] for row in rows))
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - started) * 1000
            rows.append(dict(chunk=index, frames=len(raw), wall_ms=wall_ms, denoising_steps=chunk.denoising_steps, **{f"{k}_ms": v for k, v in chunk.timings_ms.items()}))
    finally:
        reader.close()
        stream.close()
    steady = [row for row in rows if row["chunk"] >= args.warmup and row["frames"] == 4]
    if not steady:
        raise SystemExit("the source ended before the warmup chunks finished; use a longer take or fewer --warmup chunks")
    wall = [row["wall_ms"] for row in steady]
    stage_names = sorted({key for row in steady for key in row if key.endswith("_ms") and key != "wall_ms"})
    fill_ms = 4000.0 / args.source_fps
    report = dict(
        config=components.config(), resolution=[WIDTH, HEIGHT], model_load_seconds=load_seconds,
        chunk=dict(first_chunk_frames=1, later_chunk_frames=4, latents_per_chunk=1, latent_shape=list(stream.shape),
                   source_file_fps=info["fps"]),
        chunks_measured=len(steady), warmup_chunks=args.warmup, first_chunk_wall_ms=rows[0]["wall_ms"],
        chunk_wall_ms=summary(wall), stage_ms={name: summary([row[name] for row in steady]) for name in stage_names},
        throughput_fps=dict(mean=4000.0 / (sum(wall) / len(wall)), at_p95_chunk=4000.0 / percentile(wall, .95)),
        streaming_latency_ms=dict(source_fps=args.source_fps, chunk_fill_ms=fill_ms, processing_p50=percentile(wall, .5),
                                  total_p50=fill_ms + percentile(wall, .5), total_p95=fill_ms + percentile(wall, .95)),
        real_time_at_source_fps=bool(4000.0 / (sum(wall) / len(wall)) >= args.source_fps),
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, rows=rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
