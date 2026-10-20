"""`CastleStream`'s interface on the RTX 5090 runtime.

Per chunk: encode the control frames, denoise, decode, hand the frames out, then commit the clean
latent to the KV history. The clean pass is only queued when the frames are handed out (it runs on
the GPU while the caller publishes), and it stops once the last layer has written its K/V. Nothing
in a chunk waits for the GPU except the frame download.

The sampling arithmetic follows `CastleStream`: the same scheduler, warped timesteps, noise
generators and seeds, x0 computed in float64, and the same uint8 rounding of the output.
"""

from importlib import import_module

import torch

from matrix_game.inference.rollout import warp_steps
from matrix_game.inference.scheduler import FlowMatchScheduler
from matrix_game.inference.sink_policy import SINK_ROPE_MODES, rope_positions_needed, rotated_sink_position
from matrix_game.inference.stream import Chunk

LATENT_SHAPE = (1, 1, 48, 30, 54)   # one latent frame [B, F, C, h, w]
CONTROL_PAD_CHANNELS = 52            # zero channels after the 48 control channels
FRAME_TOKENS = 15 * 27               # tokens per latent frame at 480x864
TIMESTEP_SHIFT = 5.0
WINDOW = 21                          # attention window in latent frames, sink included


def load_transformer(checkpoint, *, sage: bool = False, device="cuda"):
    """Load the DiT from a release checkpoint, quantize its block linears and wrap it in the runtime."""
    from .base_model import load_causal_wan
    from .fp8_linear import quantize_block_linears

    model = load_causal_wan(str(checkpoint), local_attn_size=21, sink_size=1, device=device, dtype=torch.bfloat16)
    quantize_block_linears(model)
    if sage:
        return import_module("matrix_game.inference.rtx5090_sage.causal_model").SageCausalWanModel(model)
    from .causal_model import FastCausalWanModel
    return FastCausalWanModel(model)


class FastStream:
    """Accept [-1, 1] RGB [T, 3, 480, 864]: 1 frame first, then 4 per chunk.

    Racing: the sink is frame zero whose keys stay a fixed distance ahead of the current block once the
    window is full (future policy). Shooters: the sink is a KV-only prefix at position zero whose keys
    follow the current block at a fixed distance once the window is full (fixed policy).
    """

    def __init__(self, transformer, vae, prompt_embeds, *, seed=0, max_latents=8192, vae_backend="taehv",
                 steps=(1000,), first_steps=(1000,), profile=False, sink_latent=None,
                 sink_mode="first_frame", sink_rope_mode="future"):
        if vae_backend not in ("wan", "taehv"):
            raise ValueError("Expected wan or taehv VAE backend")
        if sink_mode not in ("kv_only", "first_frame") or sink_rope_mode not in SINK_ROPE_MODES:
            raise ValueError("Invalid sink policy")
        if max_latents < 1:
            raise ValueError("max_latents must be positive")
        self.transformer, self.vae, self.profile, self.vae_backend = transformer, vae, profile, vae_backend
        self.device, self.dtype = torch.device("cuda"), torch.bfloat16
        self.max_latents = max_latents
        self.shape = LATENT_SHAPE
        self.sink_mode, self.sink_rope_mode = sink_mode, sink_rope_mode
        if sink_latent is None:
            raise ValueError("A sink latent is required; the generated first frame is not used as the sink")
        self.first_frame_latent = sink_latent if sink_mode == "first_frame" else None
        self.sink_offset = int(sink_mode == "kv_only")
        transformer.ensure_rope_positions(rope_positions_needed(sink_rope_mode, max(8192, max_latents + self.sink_offset)))
        self.scheduler = FlowMatchScheduler(shift=TIMESTEP_SHIFT, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.steps = warp_steps(self.scheduler, list(steps))
        self.steps_first = warp_steps(self.scheduler, list(first_steps))
        self.initial_rng = torch.Generator("cpu").manual_seed(seed + 1)
        self.racing_rng = torch.Generator("cpu").manual_seed(seed)
        self.renoise_rng = torch.Generator(self.device).manual_seed(seed)
        self.kv = transformer.init_kv_cache(1, FRAME_TOKENS, self.device, self.dtype)
        self.xc = transformer.init_crossattn_cache(1, self.device, self.dtype)
        self.set_prompt(prompt_embeds)
        self.index = self.frame = 0
        self.failed = self.closed = False
        if self.sink_offset:
            # Position zero holds the sink as clean history only; it is neither rendered nor controlled.
            with torch.no_grad():
                self._forward(sink_latent.to(self.device, self.dtype), torch.zeros((), device=self.device), None, 0)
        self.encoder_features, self.decoder = [], None
        if vae_backend == "wan":
            from matrix_game.inference.decoder import IncrementalDecoder
            from matrix_game.inference.wan.modules.vae2_2 import count_conv3d
            self.encoder_features = [None] * count_conv3d(vae.model.encoder)
            self.decoder = IncrementalDecoder(vae)
        else:
            vae.clear_cache()
            if hasattr(vae, "prime"):
                vae.prime()

    def set_prompt(self, embeddings):
        """Change text conditioning at a chunk boundary without clearing visual history."""
        context = embeddings.to(self.device, self.dtype)
        if context.ndim != 3 or context.shape[1] != 512:
            raise ValueError("prompt embeddings must be [1, 512, 4096]")
        self.context = context
        for cross in self.xc:
            cross["is_init"] = False

    def _encode(self, x, param):
        """Control pixels [1, 3, T, H, W] -> one normalised latent frame [1, 1, 48, h, w], as CastleStream."""
        if self.vae_backend == "taehv":
            # TAEHV already uses the normalized Wan2.2 latent space.
            return self.vae.encode_stream(x).latent_dist.mode().permute(0, 2, 1, 3, 4)
        from matrix_game.inference.wan.modules.vae2_2 import patchify
        model = self.vae.model
        hidden = model.encoder(patchify(x, patch_size=2), feat_cache=self.encoder_features, feat_idx=[0])
        mu = model.conv1(hidden).chunk(2, dim=1)[0]
        mean, inv_std = self.vae._scale(param.device, param.dtype)
        return ((mu - mean.view(1, 48, 1, 1, 1)) * inv_std.view(1, 48, 1, 1, 1)).permute(0, 2, 1, 3, 4)

    def _sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Sigma of the scheduler grid point nearest to `timestep` (0 for a clean input), as in the wrapper."""
        timesteps = self.scheduler.timesteps.double().to(self.device)
        sigma = self.scheduler.sigmas.double().to(self.device)[torch.argmin((timesteps - timestep.double()).abs())]
        return torch.where(timestep.double() == 0, torch.zeros_like(sigma), sigma)

    # Host-side input handling, overridable by runtimes that keep the host ahead of the GPU.

    def _valid_pixels(self, pixels) -> bool:
        return bool(pixels.is_floating_point() and torch.isfinite(pixels).all()
                    and pixels.min() >= -1 and pixels.max() <= 1)

    def _upload_pixels(self, pixels, param) -> torch.Tensor:
        """RGB [T, 3, H, W] -> [1, 3, T, H, W] on the VAE's device and dtype."""
        return pixels.permute(1, 0, 2, 3).unsqueeze(0).to(param.device, param.dtype)

    def _upload_noise(self, noise) -> torch.Tensor:
        return noise.to(self.device, self.dtype)

    def _forward(self, latent, timestep, control, position, kv_only=False):
        """latent [1, 1, 48, h, w] (B, F, C), control [1, 1, 48, h, w] or None -> flow [1, 1, 48, h, w]."""
        x = latent.permute(0, 2, 1, 3, 4)
        b, _, f, h, w = x.shape
        y = x.new_zeros(b, 48 + CONTROL_PAD_CHANNELS, f, h, w)
        if control is not None:
            y[:, :48] = control.permute(0, 2, 1, 3, 4)
        t = timestep.reshape(1, 1).to(self.device, torch.float32)
        sink_position = rotated_sink_position(self.sink_rope_mode, position, WINDOW)
        with torch.autocast("cuda", dtype=self.dtype):
            flow = self.transformer(x, t, self.context, y, self.kv, self.xc, position * FRAME_TOKENS, kv_only=kv_only,
                                    sink_position=sink_position)
        return None if flow is None else flow.permute(0, 2, 1, 3, 4)

    @torch.no_grad()
    def push(self, pixels, *, frame_start, initial_noise=None, on_rgb=None):
        """Same contract as `CastleStream.push`; `on_rgb` receives owned pinned uint8 THWC frames."""
        if self.closed or self.failed:
            raise RuntimeError("Session is closed or failed; create a fresh session")
        count = 1 if self.index == 0 else 4
        if frame_start != self.frame or tuple(pixels.shape) != (count, 3, 480, 864):
            raise ValueError("Expected consecutive RGB frames: 1 initially, then 4 per chunk")
        if self.index >= self.max_latents:
            raise ValueError("Session reached its configured RoPE/frame budget")
        if not self._valid_pixels(pixels):
            raise ValueError("RGB input must be finite floating point in [-1,1]")
        if initial_noise is not None and (tuple(initial_noise.shape) != self.shape or not torch.isfinite(initial_noise).all()):
            raise ValueError("Initial noise must be finite and match one latent frame")
        try:
            events = []

            def mark():
                if self.profile:
                    events.append(torch.cuda.Event(enable_timing=True))
                    events[-1].record()

            mark()
            param = next(self.vae.parameters())
            x = self._upload_pixels(pixels, param)
            control = self._encode(x, param)
            mark()
            if tuple(control.shape) != self.shape:
                raise RuntimeError("VAE encoder output cadence changed")

            position = self.index + self.sink_offset
            if self.sink_mode == "first_frame":
                steps = self.steps_first if self.index == 0 else self.steps
            else:
                steps = self.steps_first if position < 2 else self.steps
            if self.index == 0 and self.first_frame_latent is not None:
                steps = ()
                pred = self.first_frame_latent.to(self.device, self.dtype).clone()
            else:
                if initial_noise is None:
                    rng = self.racing_rng if self.sink_mode == "first_frame" and self.index > 0 else self.initial_rng
                    initial_noise = torch.randn(self.shape, generator=rng)
                latent = self._upload_noise(initial_noise)
                for k, step in enumerate(steps):
                    timestep = torch.full((), float(step), device=self.device)
                    flow = self._forward(latent, timestep, control, position)
                    pred = (latent.double() - self._sigma(timestep) * flow.double()).to(self.dtype)
                    if k + 1 < len(steps):
                        flat = pred.flatten(0, 1).float()
                        noise = torch.randn(flat.shape, dtype=flat.dtype, device=flat.device, generator=self.renoise_rng)
                        latent = self.scheduler.add_noise(flat, noise, torch.full((1,), float(steps[k + 1]), device=self.device))
                        latent = latent.unflatten(0, (1, 1)).to(self.dtype)
            mark()

            if self.vae_backend == "wan":
                decoded = self.decoder.push(pred)
            else:
                decoded = self.vae.decode_stream(pred.permute(0, 2, 1, 3, 4)).sample[0].permute(1, 0, 2, 3)
            if tuple(decoded.shape) != (count, 3, 480, 864):
                raise RuntimeError("VAE decoder output cadence changed")
            mark()
            # Start the download, queue the clean pass behind it, then wait for the download only.
            if on_rgb is not None:
                packed = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
                host = torch.empty(packed.shape, dtype=torch.uint8, pin_memory=True)  # owned by the caller
            else:
                packed = decoded.float()
                host = torch.empty(packed.shape, dtype=torch.float32, pin_memory=True)
            host.copy_(packed, non_blocking=True)
            copied = torch.cuda.Event()
            copied.record()
            mark()
            self._forward(pred, torch.zeros((), device=self.device), None, position, kv_only=True)
            mark()
            copied.synchronize()
            rgb = None
            if on_rgb is not None:
                rgb, decoded = host, None
                on_rgb(rgb)
            else:
                decoded = host
            if self.profile:
                events[-1].synchronize()
                names = ("encode", "denoise", "decode", "pack_copy", "clean_kv")
                timings = {name: events[i].elapsed_time(events[i + 1]) for i, name in enumerate(names)}
            else:
                timings = {}
            result = Chunk(self.index, self.frame, self.frame + count - 1,
                           decoded, pred, control, len(steps), timings, rgb)
            self.index += 1
            self.frame += count
            return result
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        self.kv, self.xc = [], []
        self.encoder_features.clear()
        if self.decoder is not None:
            self.decoder.feat.clear()
        if self.vae_backend == "taehv":
            self.vae.clear_cache()
        self.closed = True


def warm_up(transformer, vae, prompt_embeds, sink_latent, sink_mode, sink_rope_mode, vae_backend, *, chunks: int = 23) -> None:
    """Run one throwaway session long enough to fill the attention window, so every kernel is
    compiled and every CUDA graph captured before the first real session (graphs and K/V buffers
    are reused by later sessions)."""
    stream = FastStream(transformer, vae, prompt_embeds, sink_latent=sink_latent, sink_mode=sink_mode,
                        sink_rope_mode=sink_rope_mode, vae_backend=vae_backend)
    try:
        for index in range(chunks):
            count = 1 if index == 0 else 4
            stream.push(torch.zeros(count, 3, 480, 864), frame_start=stream.frame)
    finally:
        stream.close()
