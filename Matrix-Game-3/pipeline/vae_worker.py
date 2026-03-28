import sys
import os

# When run as `python3 pipeline/vae_worker.py`, the parent package is unknown; add repo root like other entrypoints.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import torch.multiprocessing as mp
import time
import json
import argparse
import numpy as np
from einops import rearrange
from utils.visualize import process_video
from pipeline.vae_config import load_vae


def denormalize_video(video):
    return np.ascontiguousarray(
        ((rearrange(video, "C T H W -> T H W C").float() + 1) * 127.5)
        .clip(0, 255)
        .numpy()
        .astype(np.uint8)
    )


def _vae_worker_queue_entry(latent_queue, ack_queue, done_event, watch_dir, device_id, metadata):
    run_vae_worker_queue(latent_queue, ack_queue, done_event, watch_dir, device_id, metadata)


def start_vae_worker_process(watch_dir, device_id, metadata, queue_maxsize=2):
    """Spawn async VAE worker; returns ``(proc, latent_queue, ack_queue, done_event)``.
    Parent only sends latents; optional ``ack_queue`` sync during warmup clips.
    ``done_event`` is set after all decodes, before final ``0.mp4`` save."""
    ctx = mp.get_context("spawn")
    latent_queue = ctx.Queue(maxsize=queue_maxsize)
    ack_queue = ctx.Queue()
    done_event = ctx.Event()
    proc = ctx.Process(
        target=_vae_worker_queue_entry,
        args=(latent_queue, ack_queue, done_event, watch_dir, device_id, metadata),
        name="vae-async-worker",
    )
    proc.start()
    return proc, latent_queue, ack_queue, done_event


def run_vae_worker_queue(latent_queue, ack_queue, done_event, watch_dir, device_id, metadata):
    # print(f"🚀 VAE Worker (queue) on GPU {device_id}, output dir {watch_dir}")
    actual_watch_dir = watch_dir
    vae = load_vae(device_id=device_id, metadata=metadata)
    compile_vae = metadata.get("compile_vae", False)

    vae_cache = [None for _ in range(32)]
    all_videos = []
    num_iterations = metadata["num_iterations"]
    warmup_iters = int(metadata.get("async_vae_warmup_iters", 0))
    vae_segment_size = int(os.environ.get("WAN_VAE_SEGMENT_SIZE", "4"))

    for clip_idx in range(num_iterations):
        data = latent_queue.get()
        assert data["clip_idx"] == clip_idx, (data["clip_idx"], clip_idx)
        with torch.no_grad():
            latent = data["latent"].to(device=vae.device, dtype=vae.dtype)
            interactive = data["interactive"]
            do_compile = compile_vae and clip_idx >= 1
            video, vae_cache = vae.stream_decode(
                latent,
                vae_cache,
                first_chunk=data["first_chunk"],
                segment_size=vae_segment_size,
                compile_decoder=do_compile,
            )

            vid_cpu = video.cpu() if video is not None else None
            if vid_cpu is not None:
                all_videos.append(vid_cpu)
            else:
                print(f"  [Error] Decoding failed for clip {clip_idx}")

        if clip_idx < warmup_iters:
            ack_queue.put(clip_idx)
        if clip_idx == num_iterations - 1:
            mouse_condition_all = data["mouse_condition_all"]
            keyboard_condition_all = data["keyboard_condition_all"]
            save_name = data["save_name"]

        if interactive:
            mouse_condition = data["mouse_condition"]
            keyboard_condition = data["keyboard_condition"]
            save_name = data["save_name"]
            _save_final_video([vid_cpu], actual_watch_dir, metadata, mouse_condition, keyboard_condition, save_name, clip_idx=clip_idx, fps=metadata.get("fps", 24))

    print(f"✅ All clips decoded.", flush=True)
    print(f"🎬 Saving final video ...", flush=True)
    _save_final_video(all_videos, actual_watch_dir, metadata, mouse_condition_all, keyboard_condition_all, save_name, fps=metadata.get("fps", 24))
    print(f"🎉 Video saved to {os.path.join(actual_watch_dir, f'{save_name}.mp4')}", flush=True)
    done_event.set()
    print(f"✨ Task complete. Exiting.", flush=True)


def _save_final_video(all_videos, actual_watch_dir, metadata, mouse_condition, keyboard_condition, save_name, clip_idx=None, fps=24):
    if len(all_videos) == 0:
        return
    concatenated_video = denormalize_video(torch.concat(all_videos, dim=2)[0])
    if clip_idx is not None:
        output_path = os.path.join(actual_watch_dir, f"{save_name}_current_iteration_{clip_idx}.mp4")
    else:
        output_path = os.path.join(actual_watch_dir, f"{save_name}.mp4")

    keyboard_all = keyboard_condition.squeeze(0).float().cpu().numpy()
    mouse_all = mouse_condition.squeeze(0).float().cpu().numpy()

    process_video(
        concatenated_video.astype(np.uint8),
        output_path,
        (keyboard_all, mouse_all),
        "assets/images/mouse.png",
        mouse_scale=0.2,
        default_frame_res=(metadata["height"], metadata["width"]),
        fps=fps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=4)
    parser.add_argument(
        "--metadata_stdin",
        action="store_true",
        help="Read metadata JSON from stdin (one object). Skips waiting for metadata.json.",
    )
    args = parser.parse_args()

    meta = None
    if args.metadata_stdin:
        meta = json.load(sys.stdin)

    run_vae_worker(args.watch_dir, args.gpu_id, metadata=meta)
