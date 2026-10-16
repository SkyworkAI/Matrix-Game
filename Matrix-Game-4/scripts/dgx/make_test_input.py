#!/usr/bin/env python3
"""Write a synthetic flat-shaded "low-poly driving" test video to feed `matrix-game render --input`.

This is NOT a recording of the game. The racing model expects a clean low-poly RGB stream from the engine; no sample recording ships with the
repository and the native game cannot be built on the DGX (no SDL2, FFmpeg or GL development libraries, no display). This clip is only a
stand-in that exercises the whole pipeline: a flat-shaded road in perspective with dashed lane lines, a car-like block ahead and flat-coloured
box buildings scrolling past, all drawn with solid polygon fills. Judge speed and memory from it, not visual quality.

Usage:   python scripts/dgx/make_test_input.py --output test_input.mp4 [--seconds 6] [--fps 24] [--width 864] [--height 480]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

SKY = (122, 170, 214)
GROUND = (86, 120, 74)
ROAD = (64, 66, 72)
LINE = (232, 232, 224)
BUILDING_COLORS = [(176, 120, 96), (196, 170, 120), (120, 132, 160), (150, 96, 110), (110, 150, 140)]


def draw_frame(width: int, height: int, t: float, speed: float) -> np.ndarray:
    horizon = int(height * 0.42)
    image = Image.new("RGB", (width, height), SKY)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, horizon, width, height], fill=GROUND)
    # road: a trapezoid from the horizon to the bottom edge
    centre = width / 2
    draw.polygon([(centre - 18, horizon), (centre + 18, horizon), (centre + width * 0.42, height), (centre - width * 0.42, height)], fill=ROAD)
    # dashed centre line, scrolling toward the camera
    offset = (t * speed) % 1.0
    for k in range(14):
        near, far = (k + offset) / 14.0, (k + offset + 0.45) / 14.0
        y0, y1 = horizon + (height - horizon) * near ** 2, horizon + (height - horizon) * min(far, 1.0) ** 2
        w0, w1 = 1 + 9 * near ** 2, 1 + 9 * min(far, 1.0) ** 2
        draw.polygon([(centre - w0, y0), (centre + w0, y0), (centre + w1, y1), (centre - w1, y1)], fill=LINE)
    # buildings on both sides, scrolling with the road
    for side in (-1, 1):
        for k in range(8):
            depth = ((k + offset) / 8.0) % 1.0
            scale = depth ** 2
            y_base = horizon + (height - horizon) * scale
            x_edge = centre + side * (22 + width * 0.40 * scale)
            building_w = 6 + width * 0.16 * scale
            building_h = 14 + height * 0.55 * scale
            x0, x1 = (x_edge, x_edge + side * building_w) if side > 0 else (x_edge + side * building_w, x_edge)
            color = BUILDING_COLORS[(k * 2 + (side > 0)) % len(BUILDING_COLORS)]
            draw.rectangle([x0, y_base - building_h, x1, y_base], fill=color)
            draw.rectangle([x0, y_base - building_h, x1, y_base - building_h + max(2, building_h * 0.08)], fill=tuple(max(0, c - 40) for c in color))
    # the player's car: a low-poly block near the bottom centre, drifting slightly left and right
    car_x = centre + np.sin(t * 1.3) * width * 0.04
    car_y = height * 0.80
    draw.polygon([(car_x - 52, car_y), (car_x + 52, car_y), (car_x + 40, car_y - 34), (car_x - 40, car_y - 34)], fill=(200, 48, 44))
    draw.polygon([(car_x - 30, car_y - 34), (car_x + 30, car_y - 34), (car_x + 22, car_y - 52), (car_x - 22, car_y - 52)], fill=(150, 36, 34))
    return np.asarray(image)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--speed", type=float, default=0.9, help="road scroll speed in lane-line periods per second")
    args = ap.parse_args()

    # the model consumes blocks of four frames after the first one, so use 4k + 1 frames
    frames = int(round(args.seconds * args.fps))
    frames = (frames // 4) * 4 + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # raw RGB frames are piped to the ffmpeg binary that ships with imageio-ffmpeg (an extra of this project)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "14", str(args.output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    for index in range(frames):
        encoder.stdin.write(draw_frame(args.width, args.height, index / args.fps, args.speed).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg failed to encode the test clip")
    print(f"wrote {args.output}: {frames} frames, {args.width}x{args.height}, {args.fps} fps, {frames / args.fps:.2f} s")


if __name__ == "__main__":
    main()
