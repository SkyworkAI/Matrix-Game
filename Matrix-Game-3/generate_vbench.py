"""
Matrix-Game-3 VBench batch inference.

Loads the pipeline once, then generates num_samples videos per VBench prompt.
Saves stats to a CSV alongside the output videos.

Usage (single GPU):
    python generate_vbench.py --ckpt_dir Matrix-Game-3.0 --vbench_output_dir out/vbench/videos

Usage (multi-GPU via torchrun, experimental):
    torchrun --nproc_per_node=2 generate_vbench.py --ckpt_dir Matrix-Game-3.0 ...
"""
import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Ensure stdout/stderr can handle Unicode on Windows (cp1252 terminals)
import io, sys
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import psutil
import torch
from PIL import Image

_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
_VBENCH_ROOT       = os.path.join(_SCRIPT_DIR, "..", "..", "VBench", "vbench2_beta_i2v")
_DEFAULT_INFO_JSON = os.path.join(_VBENCH_ROOT, "vbench2_i2v_full_info.json")
_DEFAULT_CROP_DIR  = os.path.join(_VBENCH_ROOT, "vbench2_beta_i2v", "data", "crop")


def _safe(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", s)[:150]


def _fmt(secs: float) -> str:
    h, m, s = int(secs // 3600), int(secs % 3600 // 60), int(secs % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Matrix-Game-3 VBench batch inference"
    )
    # Model
    parser.add_argument("--ckpt_dir", type=str, required=True,
        help="Path to Matrix-Game-3.0 checkpoint directory.")
    parser.add_argument("--use_base_model", action="store_true", default=False,
        help="Use base (50-step) model instead of distilled (3-step).")
    parser.add_argument("--use_int8", action="store_true", default=True,
        help="Load DiT in int8 quantization (default: True).")
    parser.add_argument("--no_use_int8", action="store_false", dest="use_int8")
    parser.add_argument("--fa_version", type=str, default="2", choices=["0", "2", "3"],
        help="Flash Attention version. Default: 2.")

    # Inference
    parser.add_argument("--num_iterations", type=int, default=12,
        help="Autoregressive iterations. Frames = 57 + (N-1)*40. Default: 12 -> 497 frames.")
    parser.add_argument("--num_inference_steps", type=int, default=3,
        help="Denoising steps per iteration. Distilled: 3, base: 50. Default: 3.")
    parser.add_argument("--fps", type=int, default=24,
        help="Output video FPS. Default: 24.")
    parser.add_argument("--sample_guide_scale", type=float, default=None,
        help="CFG scale. None -> use config default.")
    parser.add_argument("--sample_shift", type=float, default=None,
        help="Flow matching shift. None -> use config default.")
    parser.add_argument("--key_strength", type=float, default=0.5,
        help="Scale applied to keyboard action conditioning. 1.0=full, 0.5=half. Default: 0.5.")
    parser.add_argument("--size", type=str, default="704*1280",
        help="Height*Width. Default: 704*1280.")

    # VAE
    parser.add_argument("--vae_type", type=str, default="mg_lightvae_v2",
        choices=["wan", "mg_lightvae", "mg_lightvae_v2"],
        help="VAE type. Default: mg_lightvae_v2 (fastest).")
    parser.add_argument("--lightvae_pruning_rate", type=float, default=0.75,
        help="LightVAE pruning rate. Default: 0.75 (v2).")

    # VBench dataset
    parser.add_argument("--vbench_info_json", type=str, default=None,
        help="Path to vbench2_i2v_full_info.json. Default: auto-detect relative to script.")
    parser.add_argument("--crop_dir", type=str, default=None,
        help="Path to VBench crop image directory.")
    parser.add_argument("--resolution", type=str, default="1-1",
        help="Crop resolution subfolder. Default: 1-1.")
    parser.add_argument("--image_types", type=str, default="scenery,indoor",
        help="Comma-separated image_type filter. Default: scenery,indoor.")

    # Output / batch control
    parser.add_argument("--vbench_output_dir", type=str, default="out/vbench/videos",
        help="Output directory for generated videos.")
    parser.add_argument("--stats_file", type=str, default=None,
        help="CSV stats path. Default: <vbench_output_dir>/../vbench_stats.csv.")
    parser.add_argument("--num_samples", type=int, default=5,
        help="Videos per prompt. Default: 5.")
    parser.add_argument("--base_seed", type=int, default=42,
        help="Base random seed. Each sample uses base_seed + sample_idx.")
    parser.add_argument("--max_prompts", type=int, default=None,
        help="Limit number of prompts (default: all).")
    parser.add_argument("--start_prompt", type=int, default=0,
        help="Start from this prompt index (0-based).")
    parser.add_argument("--skip_existing", action="store_true", default=True,
        help="Skip videos that already exist on disk (default: True).")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    parser.add_argument("--compile_vae", action="store_true", default=False,
        help="torch.compile the VAE decoder for faster decode (15-30%% speedup, one-time compile cost).")
    # ---- WorldCache args ----
    parser.add_argument("--worldcache", action="store_true", default=False,
        help="Enable WorldCache denoising-step caching (disabled by default).")
    parser.add_argument("--worldcache_thresh", type=float, default=0.40,
        help="WorldCache skip threshold (relative delta of consecutive noise predictions). Default: 0.40.")
    parser.add_argument("--worldcache_warmup", type=int, default=1,
        help="Steps before WorldCache skipping starts per clip. Default: 1.")

    args = parser.parse_args()

    # Resolve VBench paths
    if args.vbench_info_json is None:
        args.vbench_info_json = _DEFAULT_INFO_JSON
    if args.crop_dir is None:
        args.crop_dir = _DEFAULT_CROP_DIR

    # Resolve guide scale and shift from config if not provided
    if args.sample_guide_scale is None or args.sample_shift is None:
        from wan.configs import WAN_CONFIGS
        cfg = WAN_CONFIGS["matrix_game3"]
        if args.sample_guide_scale is None:
            args.sample_guide_scale = cfg.sample_guide_scale
        if args.sample_shift is None:
            args.sample_shift = cfg.sample_shift

    return args


class _PipeArgs:
    """Minimal args namespace forwarded into MatrixGame3Pipeline.generate()."""
    def __init__(self, args, save_name: str, output_dir: str):
        self.size                  = args.size
        self.num_iterations        = args.num_iterations
        self.use_int8              = args.use_int8
        self.verify_quant          = False
        self.vae_type              = args.vae_type
        self.lightvae_pruning_rate = args.lightvae_pruning_rate
        self.compile_vae           = getattr(args, 'compile_vae', False)
        self.use_async_vae         = False
        self.async_vae_warmup_iters = 0
        self.fa_version            = args.fa_version
        self.ckpt_dir              = args.ckpt_dir
        self.output_dir            = output_dir
        self.save_name             = save_name
        self.key_strength          = getattr(args, 'key_strength', 1.0)
        self.worldcache            = getattr(args, 'worldcache', False)
        self.worldcache_thresh     = getattr(args, 'worldcache_thresh', 0.40)
        self.worldcache_warmup     = getattr(args, 'worldcache_warmup', 1)
        self.fps                   = getattr(args, 'fps', 24)
        self.t5_fsdp               = False
        self.dit_fsdp              = False
        self.ulysses_size          = 1


def vbench_batch(args):
    info_json = os.path.abspath(args.vbench_info_json)
    crop_base = os.path.abspath(args.crop_dir)
    image_dir = os.path.join(crop_base, args.resolution)
    out_dir   = os.path.abspath(args.vbench_output_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats_path = (
        os.path.abspath(args.stats_file) if args.stats_file
        else os.path.join(os.path.dirname(out_dir), "vbench_stats.csv")
    )
    write_header = not os.path.isfile(stats_path) or os.path.getsize(stats_path) == 0
    stats_f = open(stats_path, "a", newline="", encoding="utf-8")
    stats_w = csv.writer(stats_f)
    if write_header:
        stats_w.writerow(["task_idx", "prompt", "sample_idx",
                          "duration_s", "gen_fps", "ram_gb", "vram_gb", "out_path", "status"])

    if not os.path.isfile(info_json):
        print(f"[vbench] ERROR: info JSON not found: {info_json}")
        stats_f.close()
        return
    if not os.path.isdir(image_dir):
        print(f"[vbench] ERROR: crop dir not found: {image_dir}")
        stats_f.close()
        return

    with open(info_json, encoding="utf-8") as f:
        entries = json.load(f)

    allowed = {t.strip() for t in args.image_types.split(",") if t.strip()} if args.image_types else None
    seen, prompts = set(), []
    for e in entries:
        name = e["image_name"]
        if name in seen:
            continue
        if allowed and e.get("image_type") not in allowed:
            continue
        seen.add(name)
        prompts.append((name, e["prompt_en"]))
    if args.start_prompt:
        prompts = prompts[args.start_prompt:]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    num_frames = 57 + (args.num_iterations - 1) * 40
    total = len(prompts) * args.num_samples
    print(f"[vbench] {len(prompts)} prompts x {args.num_samples} samples = {total} total")
    print(f"[vbench] num_iterations={args.num_iterations}  frames/video={num_frames}")
    print(f"[vbench] output -> {out_dir}")
    print(f"[vbench] stats  -> {stats_path}")

    # Load pipeline once
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from pipeline.inference_pipeline import MatrixGame3Pipeline

    cfg = WAN_CONFIGS["matrix_game3"]

    if torch.cuda.is_available():
        free_gb, total_gb = [x / 1024 ** 3 for x in torch.cuda.mem_get_info()]
        print(f"[vbench] VRAM before model load: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

    _init_args = _PipeArgs(args, save_name="init", output_dir=out_dir)
    print(f"[vbench] loading MatrixGame3Pipeline from {args.ckpt_dir} ...")
    try:
        pipeline = MatrixGame3Pipeline(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=0,
            args=_init_args,
            fa_version=args.fa_version,
            use_base_model=args.use_base_model,
        )
    except Exception as exc:
        print(f"[vbench] FATAL: model load failed -- {type(exc).__name__}: {exc}")
        traceback.print_exc()
        stats_f.close()
        return

    if torch.cuda.is_available():
        free_gb, total_gb = [x / 1024 ** 3 for x in torch.cuda.mem_get_info()]
        print(f"[vbench] VRAM after  model load: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

    skipped = generated = errors = 0
    done = 0
    ok_total_s = 0.0
    t_run_start = time.time()

    for task_idx, (image_name, prompt) in enumerate(prompts):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            print(f"[vbench] skip task {task_idx}: image not found -- {image_path}")
            continue

        img = Image.open(image_path).convert("RGB")

        for sample_idx in range(args.num_samples):
            seed = args.base_seed + task_idx * args.num_samples + sample_idx
            safe_prompt = _safe(prompt)
            out_path = os.path.join(out_dir, f"{safe_prompt}-{sample_idx}.mp4")

            if args.skip_existing and os.path.exists(out_path):
                skipped += 1
                done += 1
                stats_w.writerow([task_idx, prompt, sample_idx, "", "", "", "", out_path, "skipped"])
                stats_f.flush()
                continue

            pct = 100 * done / total if total else 0
            elapsed_run = time.time() - t_run_start
            eta = avg = ""
            if generated > 0:
                avg_s = ok_total_s / generated
                eta = f"  ETA {_fmt(avg_s * (total - done))}"
                avg = f"  avg {avg_s / 60:.1f}min/video"
            vram_free = torch.cuda.mem_get_info()[0] / 1024 ** 3 if torch.cuda.is_available() else 0.0
            print(f"\n[vbench] [{done + 1}/{total}  {pct:.0f}%{eta}{avg}]  elapsed {_fmt(elapsed_run)}")
            print(f"[vbench] task {task_idx + 1}/{len(prompts)}  sample {sample_idx + 1}/{args.num_samples}  seed {seed}  VRAM free {vram_free:.1f} GB")
            print(f"[vbench]   image : {image_name}")
            print(f"[vbench]   prompt: {prompt[:120]}")
            print(f"[vbench]   out   : {os.path.basename(out_path)}")

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            pipe_args = _PipeArgs(args, save_name=safe_prompt + f"-{sample_idx}", output_dir=out_dir)
            # Update pipeline's output_dir for this call
            pipeline.output_dir = out_dir

            try:
                t0 = time.time()
                pipeline.generate(
                    prompt,
                    img,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    shift=args.sample_shift,
                    num_inference_steps=args.num_inference_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=seed,
                    use_base_model=args.use_base_model,
                    args=pipe_args,
                )
            except SystemExit:
                pass  # pipeline calls exit() after saving -- this is expected
            except Exception as exc:
                ram_gb  = psutil.virtual_memory().used / 1024 ** 3
                vram_gb = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
                print(f"[vbench]   ERROR task {task_idx} sample {sample_idx}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                stats_w.writerow([task_idx, prompt, sample_idx, "", "", f"{ram_gb:.2f}", f"{vram_gb:.2f}",
                                   out_path, f"error:{type(exc).__name__}"])
                stats_f.flush()
                errors += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc; gc.collect()
                done += 1
                continue

            elapsed = time.time() - t0
            gen_fps  = num_frames / elapsed if elapsed > 0 else 0.0
            ram_gb   = psutil.virtual_memory().used / 1024 ** 3
            vram_gb  = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
            vram_pk  = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0

            status = "ok" if os.path.exists(out_path) else "ok_nofile"
            print(f"[vbench]   {status}  {elapsed:.1f}s  {gen_fps:.2f} gen-fps  VRAM {vram_gb:.1f} GB (peak {vram_pk:.1f} GB)  RAM {ram_gb:.1f} GB")
            print(f"[vbench]   saved -> {out_path}")
            stats_w.writerow([task_idx, prompt, sample_idx,
                               f"{elapsed:.2f}", f"{gen_fps:.2f}",
                               f"{ram_gb:.2f}", f"{vram_gb:.2f}", out_path, status])
            stats_f.flush()
            ok_total_s += elapsed
            generated  += 1
            done       += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc; gc.collect()

    elapsed_total = time.time() - t_run_start
    stats_f.close()
    print(f"\n[vbench] done -- generated={generated}  skipped={skipped}  errors={errors}  elapsed={_fmt(elapsed_total)}")
    if generated:
        print(f"[vbench] avg per video: {ok_total_s / generated / 60:.1f} min  ({ok_total_s / generated:.1f}s)")
    print(f"[vbench] stats -> {stats_path}")


if __name__ == "__main__":
    args = _parse_args()
    vbench_batch(args)
