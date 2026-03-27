import sys
import os
import argparse
import logging
import warnings
import random
import torch
import torch.distributed as dist
warnings.filterwarnings('ignore')
from PIL import Image
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from pipeline.inference_interactive_pipeline import MatrixGame3Pipeline as MatrixGame3InteractivePipeline
from pipeline.inference_pipeline import MatrixGame3Pipeline
from utils.misc import set_seed
def _validate_args(args):
    if args.ulysses_size <= 1:
        if args.t5_fsdp or args.dit_fsdp:
            logging.info(f"Single GPU detected (ulysses_size={args.ulysses_size}). Automatically disabling FSDP (t5_fsdp and dit_fsdp).")
            args.t5_fsdp = False
            args.dit_fsdp = False

    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.prompt is not None, "please specify the prompt."
    assert args.image is not None, "Please specify the image."

    cfg = WAN_CONFIGS["matrix_game3"]

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    args.seed = args.seed if args.seed >= 0 else random.randint(
        0, sys.maxsize)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from an image, a text prompt and corresponding action path using matrix-game-3."
    )
    # Core runtime
    parser.add_argument("--size", type=str, default="1280*704")
    parser.add_argument("--ckpt_dir",type=str,default=None,help="The path to the checkpoint directory.")

    # Distributed / parallel
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")

    # Input conditioning
    parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the video from.")
    parser.add_argument("--seed", type=int, default=42, help="The seed to use for generating the video.")
    parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")
    parser.add_argument('--num_iterations', type=int, default=12)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False, help="Whether to convert model paramerters dtype.")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--save_name', type=str, default="generated_video")
    # Sampling parameters
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="Classifier free guidance scale.")
    parser.add_argument('--num_inference_steps', type=int, default=50)

    # VAE
    parser.add_argument('--lightvae_pruning_rate', type=float, default=None, help='Pruning rate for lightvae_wan22. If unset, infer dynamically from checkpoint.')
    parser.add_argument('--use_async_vae', action='store_true', help='Whether to use asynchronous VAE decoding.')
    parser.add_argument('--async_vae_warmup_iters', type=int, default=0, help='Number of initial iterations to force sync VAE for warmup in async mode.')
    parser.add_argument('--compile_vae', action='store_true', help='Whether to use torch.compile for VAE decoder.')
    parser.add_argument('--vae_type', type=str, default='mg_lightvae_v2', choices=['wan', 'mg_lightvae', 'mg_lightvae_v2'], help='VAE type.')
    
    # Quantization
    parser.add_argument('--use_int8', action='store_true', help='Whether to use int8 quantization for DiT.')
    parser.add_argument('--verify_quant', action='store_true', help='Whether to verify quantization accuracy by comparing with BF16.')
    
    parser.add_argument('--fa_version', type=str, default=None, choices=['0', '2', '3'], help='Flash Attention version (2 or 3). Set to 0 to disable.')
    parser.add_argument("--interactive", action="store_true", help="Enable interactive inference.")
    parser.add_argument("--use_base_model", action="store_true", help="Enable base model inference.")
    parser.add_argument("--worldcache", action="store_true", default=False,
        help="Enable WorldCache denoising-step caching (disabled by default).")
    parser.add_argument("--worldcache_thresh", type=float, default=0.40,
        help="WorldCache skip threshold. Default: 0.40.")
    parser.add_argument("--worldcache_warmup", type=int, default=1,
        help="Steps before WorldCache skipping starts per clip. Default: 1.")
    args = parser.parse_args()
    _validate_args(args)
    return args

def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def generate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)
    set_seed(args.seed)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    cfg = WAN_CONFIGS["matrix_game3"]
    if args.ulysses_size > 1 and cfg.num_heads % args.ulysses_size != 0 and rank == 0:
        logging.warning(
            f"`{cfg.num_heads=}` is not divisible by `{args.ulysses_size=}`. "
            "Ulysses attention will pad heads internally and trim back after communication.")

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        seed = [args.seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed, src=0)
        args.seed = seed[0]

    logging.info(f"Input image: {args.image}")
    pil_image = Image.open(args.image).convert("RGB")

    logging.info(f"Input prompt: {args.prompt}")
    logging.info("Creating Matrix-Game-3 pipeline.")

    if args.interactive:
        pipeline = MatrixGame3InteractivePipeline(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            args=args,
            fa_version=args.fa_version,
            use_base_model=args.use_base_model,
        )
    else:
        pipeline = MatrixGame3Pipeline(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            args=args,
            fa_version=args.fa_version,
            use_base_model=args.use_base_model,
        )

    logging.info("Generating video ...")
    pipeline.generate(
        args.prompt,
        pil_image,
        max_area=MAX_AREA_CONFIGS[args.size],
        shift=args.sample_shift,
        num_inference_steps=args.num_inference_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.seed,
        use_base_model=args.use_base_model,
        args=args)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
