"""Scene-configured chunk1 sessions with Wan or TAEHV streaming codecs."""
from dataclasses import dataclass

import torch


@torch.no_grad()
def encode_sink_image(vae, path, vae_backend, *, first_frame=False):
    """Encode an independent image, without seeding either streaming VAE cache."""
    from PIL import Image, ImageOps
    from matrix_game.inference.preprocessing import crop_and_resize, frames_to_tensor

    if vae_backend not in ("wan", "taehv"):
        raise ValueError("Expected wan or taehv VAE backend")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = (ImageOps.fit(image, (864, 480), method=Image.Resampling.LANCZOS) if first_frame
                 else crop_and_resize(image, 480, 864))
        pixels = frames_to_tensor([image])
    param = next(vae.parameters())
    pixels = pixels.unsqueeze(0).to(param.device, param.dtype)
    if vae_backend == "wan":
        latent = vae.encode_to_latent(pixels)
    else:
        vae.clear_cache()
        try:
            latent = vae.encode(pixels.permute(0, 2, 1, 3, 4)).latent_dist.mode().permute(0, 2, 1, 3, 4)
        finally:
            vae.clear_cache()
    if tuple(latent.shape) != (1, 1, 48, 30, 54) or not torch.isfinite(latent).all():
        raise ValueError("Sink image must encode to one finite 48-channel latent")
    return latent.detach()


@dataclass
class Chunk:
    index: int
    frame_start: int
    frame_end: int
    pixels: torch.Tensor
    latent: torch.Tensor
    control: torch.Tensor
    denoising_steps: int
    timings_ms: dict
    rgb: torch.Tensor = None


class CastleStream:
    """Accept [-1,1] RGB [T,3,480,864]; retain only bounded inference caches.

    Shooter sinks are KV-only; racing sinks become output frame zero. Neither
    is used as low-poly control input. No retrieved memory, preview or resets.
    Initial noise may be supplied explicitly for native-rollout equivalence tests.
    """

    def __init__(self, generator, vae, prompt_embeds, *, seed=0, max_latents=8192,
                 vae_backend="wan", steps=(1000, 500), first_steps=(1000, 750, 500, 250), profile=False,
                 sink_latent=None, sink_mode="kv_only", sink_rope_mode="fixed"):
        from matrix_game.inference.rollout import CausalV2VRollout, set_attention_window, warp_steps
        from matrix_game.inference.wrappers import frame_seq_length_of

        self.gen, self.vae, self.prompt = generator, vae, prompt_embeds
        from matrix_game.inference.sink_policy import SINK_ROPE_MODES, rope_positions_needed
        if sink_mode not in ("kv_only", "first_frame") or sink_rope_mode not in SINK_ROPE_MODES:
            raise ValueError("Invalid sink policy")
        self.sink_mode, self.sink_rope_mode = sink_mode, sink_rope_mode
        self.first_frame_latent = sink_latent if sink_mode == "first_frame" else None
        if vae_backend not in ("wan", "taehv"):
            raise ValueError("Expected wan or taehv VAE backend")
        for values in (steps, first_steps):
            if not values or values[0] != 1000 or any(not 0 < v <= 1000 for v in values) or any(a <= b for a, b in zip(values, values[1:])):
                raise ValueError("Denoising timesteps must decrease strictly from 1000")
        self.vae_backend, self.profile = vae_backend, profile
        param = next(generator.parameters())
        self.device, self.dtype = param.device, param.dtype
        self.shape = (1, 1, 48, 30, 54)
        if sink_latent is None:
            raise ValueError("A sink latent is required; the generated first frame is not used as the sink")
        if (tuple(sink_latent.shape) != self.shape
                or not sink_latent.is_floating_point() or not torch.isfinite(sink_latent).all()):
            raise ValueError("Sink must be one finite floating-point latent frame")
        self.sink_offset = int(sink_mode == "kv_only")
        self.index = self.frame = 0
        self.failed = self.closed = False
        self.max_latents = max_latents
        if max_latents < 1:
            raise ValueError("max_latents must be positive")
        self.initial_rng = torch.Generator("cpu").manual_seed(seed + 1)
        self.racing_rng = torch.Generator("cpu").manual_seed(seed)
        self.renoise_rng = torch.Generator(self.device).manual_seed(seed)
        scheduler = generator.get_scheduler()
        self.rollout = CausalV2VRollout(
            generator, scheduler, warp_steps(scheduler, list(steps)),
            warp_steps(scheduler, list(first_steps)),
            local_attn_size=21, sink_size=1, sink_rope_mode=sink_rope_mode,
            memory_mode="none",
            rope_max_len=rope_positions_needed(sink_rope_mode, max(8192, max_latents + self.sink_offset)),
        )
        set_attention_window(generator, 21, 1)
        self.rollout._prepare_rope()
        self.fsl = frame_seq_length_of(generator, 30, 54)
        self.kv, self.xc = self.rollout._caches(1, 21, self.fsl, self.dtype, self.device)
        for cache in self.kv:
            cache["k_sink_raw"] = torch.zeros_like(cache["k"][:, :self.fsl])
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
        runtime = getattr(getattr(generator, "model", None), "castle_runtime", None)
        if runtime is not None:
            runtime.prime(generator, self.prompt, self.shape, self.kv, self.xc, self.fsl, sink_rope_mode)
        if self.sink_offset:
            # Position zero is reserved for identity reference, not a gameplay frame.
            with torch.no_grad():
                self.gen(noisy_image_or_video=sink_latent.to(self.device, self.dtype),
                         conditional_dict={"prompt_embeds": self.prompt},
                         timestep=torch.zeros(1, 1, device=self.device), drop_control=True,
                         kv_cache=self.kv, crossattn_cache=self.xc, current_start=0,
                         sink_rope_mode=self.sink_rope_mode)

    def set_prompt(self, embeddings):
        """Change text conditioning at a chunk boundary without clearing visual history."""
        if self.closed or self.failed or embeddings.shape != self.prompt.shape:
            raise ValueError("Prompt update requires an active session and unchanged token shape")
        self.prompt = embeddings
        for cross in self.xc:
            cross["is_init"] = False

    @torch.no_grad()
    def push(self, pixels, *, frame_start, initial_noise=None, on_rgb=None):
        """Optionally publish owned CPU uint8 THWC frames before clean-KV finishes.

        A callback replaces the CPU float output. The call still waits for clean
        KV, so throughput accounting and next-chunk causality stay unchanged.
        """
        if self.closed or self.failed:
            raise RuntimeError("Session is closed or failed; create a fresh session")
        count = 1 if self.index == 0 else 4
        if frame_start != self.frame or tuple(pixels.shape) != (count, 3, 480, 864):
            raise ValueError("Expected consecutive RGB frames: 1 initially, then 4 per chunk")
        if self.index >= self.max_latents:
            raise ValueError("Session reached its configured RoPE/frame budget")
        if not pixels.is_floating_point() or not torch.isfinite(pixels).all() or pixels.min() < -1 or pixels.max() > 1:
            raise ValueError("RGB input must be finite floating point in [-1,1]")
        if initial_noise is not None and (tuple(initial_noise.shape) != self.shape or not torch.isfinite(initial_noise).all()):
            raise ValueError("Initial noise must be finite and match one latent frame")
        try:
            events = []
            def mark():
                if self.profile:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    events.append(event)
            mark()
            param = next(self.vae.parameters())
            x = pixels.permute(1, 0, 2, 3).unsqueeze(0).to(param.device, param.dtype)
            if self.vae_backend == "wan":
                from matrix_game.inference.wan.modules.vae2_2 import patchify
                model = self.vae.model
                hidden = model.encoder(patchify(x, patch_size=2), feat_cache=self.encoder_features, feat_idx=[0])
                mu = model.conv1(hidden).chunk(2, dim=1)[0]
                mean, inv_std = self.vae._scale(param.device, param.dtype)
                control = ((mu - mean.view(1, 48, 1, 1, 1)) * inv_std.view(1, 48, 1, 1, 1)).permute(0, 2, 1, 3, 4)
            else:
                # TAEHV already uses the normalized Wan2.2 latent space.
                control = self.vae.encode_stream(x).latent_dist.mode().permute(0, 2, 1, 3, 4)
            mark()
            if tuple(control.shape) != self.shape:
                raise RuntimeError("VAE encoder output cadence changed")
            position = self.index + self.sink_offset
            if self.sink_mode == "first_frame":
                steps = self.rollout.steps_first if self.index == 0 else self.rollout.steps
            else:
                steps = self.rollout.steps_first if position < 2 else self.rollout.steps
            reference_first = self.index == 0 and self.first_frame_latent is not None
            if reference_first:
                steps = ()
                pred = self.first_frame_latent.to(self.device, self.dtype).clone()
            else:
                if initial_noise is None:
                    rng = self.racing_rng if self.sink_mode == "first_frame" and self.index > 0 else self.initial_rng
                    initial_noise = torch.randn(self.shape, generator=rng)
                x = initial_noise.to(self.device, self.dtype)
            cond = {"prompt_embeds": self.prompt, "control_latent": control}
            common = dict(kv_cache=self.kv, crossattn_cache=self.xc,
                          current_start=position * self.fsl, sink_rope_mode=self.sink_rope_mode)
            for k, step in enumerate(steps):
                t = torch.full((1, 1), float(step), device=self.device)
                _, pred = self.gen(noisy_image_or_video=x, conditional_dict=cond, timestep=t, **common)
                if k + 1 < len(steps):
                    flat = pred.flatten(0, 1).float()
                    noise = torch.randn(flat.shape, dtype=flat.dtype, device=flat.device, generator=self.renoise_rng)
                    x = self.rollout.scheduler.add_noise(flat, noise, torch.full((1,), float(steps[k + 1]), device=self.device))
                    x = x.unflatten(0, (1, 1)).to(self.dtype)
            if not torch.isfinite(pred).all():
                raise RuntimeError("Non-finite generated latent")
            mark()
            def clean():
                self.gen(noisy_image_or_video=pred, conditional_dict=cond,
                         timestep=torch.zeros(1, 1, device=self.device), drop_control=True, **common)

            if on_rgb is None:
                clean()
                mark()
            if self.vae_backend == "wan":
                decoded = self.decoder.push(pred)
            else:
                decoded = self.vae.decode_stream(pred.permute(0, 2, 1, 3, 4)).sample[0].permute(1, 0, 2, 3)
            mark()
            if tuple(decoded.shape) != (count, 3, 480, 864):
                raise RuntimeError("VAE decoder output cadence changed")
            rgb = None
            if on_rgb is not None:
                packed = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
                if packed.is_cuda:
                    # Each publication owns its storage, never a reused graph/pinned buffer.
                    rgb = torch.empty(packed.shape, dtype=torch.uint8, pin_memory=True)
                    rgb.copy_(packed, non_blocking=True)
                    copied = torch.cuda.Event()
                    copied.record()
                    mark()
                    copied.synchronize()
                    try:
                        on_rgb(rgb)
                    finally:
                        # The native wrapper has host syncs. Publish before
                        # entering it, rather than merely before its final wait.
                        clean()
                        mark()
                        torch.cuda.current_stream(self.device).synchronize()
                else:
                    rgb = packed.cpu()
                    on_rgb(rgb)
                    mark()
                    clean()
                    mark()
                decoded = None
            else:
                decoded = decoded.cpu()
            if self.profile:
                events[-1].synchronize()
            names = ("encode", "denoise", "clean_kv", "decode") if on_rgb is None else (
                "encode", "denoise", "decode", "pack_copy", "clean_kv")
            timings = {name: events[i].elapsed_time(events[i + 1]) for i, name in enumerate(names)} if self.profile else {}
            result = Chunk(self.index, self.frame, self.frame + count - 1,
                           decoded, pred.detach(), control.detach(), len(steps), timings, rgb)
            self.index += 1
            self.frame += count
            return result
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.closed:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        runtime = getattr(getattr(self.gen, "model", None), "castle_runtime", None)
        if runtime is not None:
            runtime.clear()
        self.kv.clear()
        self.xc.clear()
        self.encoder_features.clear()
        if self.decoder is not None:
            self.decoder.feat.clear()
        if self.vae_backend == "taehv":
            self.vae.clear_cache()
        self.closed = True
