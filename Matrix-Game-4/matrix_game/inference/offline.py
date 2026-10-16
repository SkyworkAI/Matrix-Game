"""Render a video or a complete local recording using the same stream as online."""
import argparse
import json
from pathlib import Path
import time

from matrix_game.inference.resolution import HEIGHT, WIDTH
from matrix_game.scenes import add_model_arguments, resolve_model_arguments, SCENES


def frame_blocks(reader):
    count = 1
    while True:
        frames = []
        for _ in range(count):
            value = next(reader, None)
            if value is None:
                break
            frames.append(value)
        if not frames:
            return
        valid = len(frames)
        frames.extend([frames[-1]] * (count - valid))
        yield frames, valid
        if valid < count:
            return
        count = 4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.set_defaults(preset="quality")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Clean low-poly RGB video")
    source.add_argument("--take", type=Path, help="Completed game recording directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--captions", type=Path, help="Dynamic mode: JSONL with one frame_start/frame_end/prompt record per chunk")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists")
    manifest = None
    if args.take:
        manifest = json.loads((args.take / "session.json").read_text())
        if manifest.get("status") != "complete" or manifest.get("frames", 0) < 1:
            parser.error("Expected a complete nonempty recording")
        scene = manifest.get("scene", manifest.get("game"))
        if args.scene == "racing" and manifest.get("mode") in ("drive", "time_trial", "pursuit"):
            scene = "racing"
        if scene not in SCENES or manifest.get("prompt") != SCENES[scene]["prompt"]:
            parser.error("Recording scene/prompt does not match release profiles")
        args.scene, args.input = scene, args.take / "reference.mkv"
    try:
        resolve_model_arguments(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.prompt_mode == "dynamic" and args.captions is None:
        parser.error("Offline dynamic mode requires --captions; an RGB video alone contains no engine captions")
    if args.prompt_mode != "dynamic" and args.captions is not None:
        parser.error("--captions requires --prompt-mode dynamic")
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image
    import torch
    from .loader import ModelComponents
    from .preprocessing import crop_and_resize, frames_to_tensor, tensor_to_uint8
    if args.dit_runtime.startswith("rtx5090"):
        from .rtx5090 import FastStream as CastleStream
    else:
        from .stream import CastleStream

    torch.manual_seed(args.seed)
    reader = imageio_ffmpeg.read_frames(str(args.input), pix_fmt="rgb24")
    writer = stream = caption_file = None
    try:
        info = next(reader)
        width, height = info["size"]
        fps = info["fps"]
        if manifest and ((width, height) != (manifest["width"], manifest["height"]) or abs(fps - manifest["fps"]) > .001):
            raise ValueError("Recording dimensions or FPS differ from manifest")
        components = ModelComponents(args)
        components.load()
        stream = CastleStream(components.gen, components.vae, components.prompt,
                              seed=args.seed, vae_backend=args.vae_backend, steps=args.steps,
                              first_steps=args.first_steps, sink_latent=components.sink_latent,
                              sink_mode=args.sink_mode, sink_rope_mode=args.sink_rope_mode)
        if args.captions:
            caption_file = args.captions.open()
        current_prompt = args.prompt
        args.output.mkdir(parents=True)
        partial = args.output / "rendered.partial.mp4"
        writer = imageio_ffmpeg.write_frames(str(partial), (WIDTH, HEIGHT), fps=fps, codec="libx264",
                    pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
                    output_params=["-crf", "16", "-preset", "fast", "-movflags", "+faststart"])
        writer.send(None)
        written, padded = 0, 0
        started = time.perf_counter()
        for index, (raw, valid) in enumerate(frame_blocks(reader)):
            if caption_file is not None:
                record = json.loads(caption_file.readline())
                if record.get("frame_start") != written or record.get("frame_end") != written + valid - 1:
                    raise ValueError("Caption ranges must match consecutive decoded RGB chunks")
                candidate = record.get("prompt")
                if not isinstance(candidate, str) or not candidate.strip() or len(candidate) > 8192:
                    raise ValueError("Invalid caption")
                if index % args.prompt_interval_chunks == 0 and candidate != current_prompt:
                    stream.set_prompt(components.caption_encoder(candidate))
                    current_prompt = candidate
            frames = [crop_and_resize(Image.fromarray(np.frombuffer(r, np.uint8).reshape(height, width, 3)), HEIGHT, WIDTH) for r in raw]
            result = stream.push(frames_to_tensor(frames), frame_start=written)
            rgb = tensor_to_uint8(result.pixels)[:valid]
            if len(rgb) != valid:
                raise ValueError("Decoder output cadence mismatch")
            for frame in rgb:
                writer.send(frame.tobytes())
            written += valid
            padded += len(raw) - valid
            if index % 25 == 0:
                print(f"{written} frames | {written / (time.perf_counter() - started):.2f} processing FPS", flush=True)
        if written == 0 or (manifest and written != manifest["frames"]):
            raise ValueError("Input frame count differs from recording manifest")
        if caption_file is not None and caption_file.read().strip():
            raise ValueError("Caption file extends beyond the input video")
        writer.close()
        writer = None
        # Never label a truncated encoder output as a completed rendering.
        check = imageio_ffmpeg.read_frames(str(partial), pix_fmt="rgb24")
        try:
            encoded = next(check)
            count = sum(1 for _ in check)
        finally:
            check.close()
        if count != written or encoded["size"] != (WIDTH, HEIGHT) or abs(encoded["fps"] - fps) > .01:
            raise RuntimeError("Encoded output failed frame-count verification")
        partial.rename(args.output / "rendered.mp4")
        report = dict(components.config(), frames=written, padded_frames=padded, fps=fps,
                      seed=args.seed, status="complete", source_name=args.input.name,
                      seconds=time.perf_counter() - started, torch=torch.__version__)
        (args.output / "render.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Render complete:", args.output / "rendered.mp4")
    finally:
        reader.close()
        if writer is not None:
            writer.close()
        if stream is not None:
            stream.close()
        if caption_file is not None:
            caption_file.close()
