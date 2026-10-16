"""Public scene profiles. Resource paths are supplied by the caller."""
import argparse
import json
from pathlib import Path

SCENES = {
    name: json.loads((Path(__file__).parent / "configs" / (name + ".json")).read_text())
    for name in ("castle", "ancient-egypt", "racing")
}


def configure_scene(args):
    profile = SCENES[args.scene]
    args.prompt = getattr(args, "prompt", None) or profile["prompt"]
    args.sink_mode = "first_frame" if args.scene == "racing" else "kv_only"
    args.sink_rope_mode = "absolute" if args.scene == "racing" else "fixed"
    root = args.resources.resolve()
    args.model_dir = args.model_dir or root / "base"
    args.checkpoint = args.checkpoint or root / "models" / args.scene / "model.safetensors"
    args.taehv_path = args.taehv_path or root / "base/taew2_2.pth"
    args.asset_root = args.asset_root or root / ("racing" if args.scene == "racing" else "game")
    if args.sink_image is None and not args.no_sink_image:
        args.sink_image = root / "models" / args.scene / "sink.png"


OPTIMIZED_RUNTIMES = ("rtx5090", "rtx5090-sage", "rtx5080", "rtx5080-sage", "rtx5070ti", "rtx5070ti-sage")
FIXED_PROMPT_RUNTIMES = ("rtx5080", "rtx5080-sage", "rtx5070ti", "rtx5070ti-sage")


def add_model_arguments(parser):
    parser.add_argument("--scene", choices=tuple(SCENES), default="ancient-egypt")
    parser.add_argument("--resources", type=Path, default=Path("resources"))
    for key in ("model-dir", "checkpoint", "taehv-path", "asset-root"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--preset", choices=("realtime", "quality", "baseline") + OPTIMIZED_RUNTIMES, default="realtime")
    parser.add_argument("--vae-backend", choices=("wan", "taehv"))
    parser.add_argument("--dit-runtime", choices=("native", "kernels", "cudagraph", "ring", "ring-graph", "ring-fused") + OPTIMIZED_RUNTIMES)
    parser.add_argument("--steps", type=parse_steps)
    parser.add_argument("--first-steps", type=parse_steps)
    parser.add_argument("--vae-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prompt-mode", choices=("fixed", "dynamic"), default="fixed")
    parser.add_argument("--prompt", help="Override the fixed scene caption")
    parser.add_argument("--prompt-interval-chunks", type=int, default=1)
    parser.add_argument("--t5-runtime", choices=("native", "graph"), default="graph")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sink-image", type=Path)
    group.add_argument("--no-sink-image", action="store_true")


def parse_steps(value):
    import argparse
    try:
        steps = tuple(int(v) for v in value.split(","))
        if not steps or steps[0] != 1000 or any(not 0 < v <= 1000 for v in steps) or any(a <= b for a, b in zip(steps, steps[1:])):
            raise ValueError
        return steps
    except ValueError:
        raise argparse.ArgumentTypeError("Timesteps must decrease strictly from 1000") from None


def resolve_model_arguments(args, require_model=True):
    configure_scene(args)
    if args.prompt_mode == "dynamic" and args.scene != "racing":
        raise ValueError("Dynamic captions are currently supported for racing only")
    if not 1 <= args.prompt_interval_chunks <= 1024:
        raise ValueError("Prompt interval must be 1..1024 chunks")
    realtime = args.preset == "realtime" or args.preset in OPTIMIZED_RUNTIMES
    if args.preset in OPTIMIZED_RUNTIMES:
        args.dit_runtime = args.dit_runtime or args.preset
    if args.dit_runtime in FIXED_PROMPT_RUNTIMES and args.prompt_mode == "dynamic":
        raise ValueError(f"The {args.dit_runtime} runtime supports fixed prompts only")
    args.vae_backend = args.vae_backend or ("taehv" if realtime else "wan")
    args.dit_runtime = args.dit_runtime or ("ring-fused" if args.preset == "realtime" else "native")
    args.steps = args.steps or ((1000,) if realtime else (1000, 500))
    args.first_steps = args.first_steps or args.steps
    if args.vae_graph is None:
        args.vae_graph = realtime and args.vae_backend == "taehv"
    if args.vae_graph and args.vae_backend != "taehv":
        raise ValueError("VAE graph requires TAEHV")
    if require_model:
        files = [args.checkpoint, args.model_dir / "models_t5_umt5-xxl-enc-bf16.pth"]
        files.append(args.taehv_path if args.vae_backend == "taehv" else args.model_dir / "Wan2.2_VAE.pth")
        if args.sink_image is not None:
            files.append(args.sink_image)
        missing = [str(p) for p in files if not p.is_file()]
        if not (args.model_dir / "google/umt5-xxl").is_dir():
            missing.append(str(args.model_dir / "google/umt5-xxl"))
        if missing:
            raise FileNotFoundError("Missing resources:\n" + "\n".join(missing))
    return args
