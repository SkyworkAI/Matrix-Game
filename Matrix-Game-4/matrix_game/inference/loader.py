"""Model ownership shared by online and offline inference."""
import gc
import hashlib
import logging
from matrix_game.scenes import SCENES

LOG = logging.getLogger(__name__)


class ModelComponents:
    def __init__(self, args):
        self.args = args
        self.gen = self.vae = self.prompt = None
        self.sink_latent = self.sink_sha256 = self.attention_backend = None
        self.caption_encoder = None

    def load(self):
        import torch
        from matrix_game.inference.loading import build_dit
        from matrix_game.inference.wrappers import WanTextEncoder, WanVAE22Wrapper
        from matrix_game.inference.wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.attention_backend = "fa3" if FLASH_ATTN_3_AVAILABLE else ("fa2" if FLASH_ATTN_2_AVAILABLE else "sdpa")
        device, dtype = torch.device("cuda"), torch.bfloat16
        self.gen = build_dit(str(self.args.model_dir), str(self.args.checkpoint), is_causal=True,
                            dtype="bfloat16", timestep_shift=5, ckpt_keys=("generator_ema",)).to(device).eval().requires_grad_(False)
        from matrix_game.inference.runtime import install_runtime
        if self.args.dit_runtime.startswith("ring"):
            LOG.warning("Ring/fused runtime changes floating-point reduction order; use native for baseline comparisons")
        install_runtime(self.gen, self.args.dit_runtime)
        text = WanTextEncoder(t5_checkpoint_path=str(self.args.model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
                              tokenizer_path=str(self.args.model_dir / "google/umt5-xxl")).to(device, dtype).eval()
        if self.args.prompt_mode == "dynamic":
            from matrix_game.inference.text import CaptionEncoder
            self.caption_encoder = CaptionEncoder(text, graph=self.args.t5_runtime == "graph")
            self.prompt = self.caption_encoder(self.args.prompt)
        else:
            with torch.no_grad():
                self.prompt = text(text_prompts=[self.args.prompt])["prompt_embeds"].to(dtype)
        del text
        gc.collect()
        torch.cuda.empty_cache()
        if self.args.vae_backend == "wan":
            self.vae = WanVAE22Wrapper(vae_path=str(self.args.model_dir / "Wan2.2_VAE.pth"))
        else:
            from matrix_game.inference.taehv import TAEHV
            self.vae = TAEHV(str(self.args.taehv_path))
        self.vae = self.vae.to(device, dtype).eval().requires_grad_(False)
        if getattr(self.args, "vae_graph", False):
            if self.args.vae_backend != "taehv":
                raise ValueError("VAE graph requires TAEHV")
            from matrix_game.inference.vae_graph import StreamingVAEGraph
            self.vae = StreamingVAEGraph(self.vae)
        if self.args.sink_image is not None:
            from matrix_game.inference.stream import encode_sink_image
            self.sink_latent = encode_sink_image(self.vae, self.args.sink_image, self.args.vae_backend,
                first_frame=self.args.sink_mode == "first_frame").to(device, dtype)
            self.sink_sha256 = hashlib.sha256(self.args.sink_image.read_bytes()).hexdigest()
        LOG.info("Model config: %s", self.config())

    def stream(self, **kwargs):
        from matrix_game.inference.stream import CastleStream
        return CastleStream(self.gen, self.vae, self.prompt, vae_backend=self.args.vae_backend,
                            steps=self.args.steps, first_steps=self.args.first_steps,
                            sink_latent=self.sink_latent, sink_mode=self.args.sink_mode,
                            sink_rope_mode=self.args.sink_rope_mode, **kwargs)

    def config(self):
        return dict(scene=self.args.scene, scene_name=SCENES[self.args.scene]["name"],
                    checkpoint=self.args.checkpoint.name, ckpt_keys=["generator_ema"],
                    vae_backend=self.args.vae_backend, steps=list(self.args.steps),
                    first_steps=list(self.args.first_steps), sink_rope_mode=self.args.sink_rope_mode,
                    sink_mode=self.args.sink_mode, sink_encoder=self.args.vae_backend,
                    sink_source="image" if self.args.sink_image is not None else "generated_first_frame",
                    sink_image=self.args.sink_image.name if self.args.sink_image is not None else None,
                    sink_sha256=self.sink_sha256, prompt=self.args.prompt,
                    hq_first_frame=self.args.sink_mode == "first_frame" and self.args.sink_image is not None,
                    prompt_mode=self.args.prompt_mode, prompt_interval_chunks=self.args.prompt_interval_chunks,
                    t5_runtime=self.args.t5_runtime, dit_parallel="single", dit_runtime=self.args.dit_runtime,
                    attention_backend=self.attention_backend,
                    vae_graph=getattr(self.args, "vae_graph", False),
                    early_publish=getattr(self.args, "early_publish", False),
                    profile_chunks=getattr(self.args, "profile_chunks", False),
                    reference_pipeline=getattr(self.args, "reference_pipeline", "stream"),
                    experimental_runtime=self.args.dit_runtime.startswith("ring"))
