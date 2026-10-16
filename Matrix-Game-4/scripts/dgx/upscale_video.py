#!/usr/bin/env python3
"""Upscale a rendered video 2x with Real-ESRGAN (x2 weights loaded through spandrel), frame by frame on the GPU.

The model renders at a fixed 864x480, so a higher resolution comes from this post-process, not from the renderer. Frames are piped to and from
the ffmpeg binary that ships with imageio-ffmpeg; the output is H.264 (yuv420p) at the source frame rate. Audio is not carried over.

Usage:   python scripts/dgx/upscale_video.py --input rendered.mp4 --output rendered_1728x960.mp4 --weights RealESRGAN_x2.pth [--crf 14]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from spandrel import ImageModelDescriptor, ModelLoader


def probe(ffmpeg: str, path: Path) -> tuple[int, int, float]:
    text = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True).stderr
    size = re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", text)
    fps = re.search(r"(\d+(?:\.\d+)?) fps", text)
    if not size or not fps:
        raise SystemExit(f"cannot read the video size and frame rate of {path}")
    return int(size.group(1)), int(size.group(2)), float(fps.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--crf", type=int, default=14)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; not overwriting")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    width, height, fps = probe(ffmpeg, args.input)
    model = ModelLoader().load_from_file(str(args.weights))
    if not isinstance(model, ImageModelDescriptor) or model.scale != 2:
        raise SystemExit(f"{args.weights} is not a 2x image model")
    model = model.cuda().eval().half()
    out_width, out_height = width * 2, height * 2

    decoder = subprocess.Popen([ffmpeg, "-loglevel", "error", "-i", str(args.input), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    encoder = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_width}x{out_height}", "-r", str(fps),
                                "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf), str(args.output)], stdin=subprocess.PIPE)
    frame_bytes = width * height * 3
    count = 0
    with torch.inference_mode():
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()).cuda()
            tensor = frame.permute(2, 0, 1).unsqueeze(0).half() / 255.0
            result = model(tensor).squeeze(0).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).contiguous().cpu().numpy()
            encoder.stdin.write(result.tobytes())
            count += 1
    encoder.stdin.close()
    if encoder.wait() != 0 or decoder.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(json.dumps({"input": str(args.input), "output": str(args.output), "frames": count, "size": [out_width, out_height], "fps": fps}))


if __name__ == "__main__":
    main()
