"""Causal Wan2.2 DiT with a rolling self-attention KV cache (inference only).

Derived from Wan (https://github.com/Wan-Video/Wan2.2) and Self-Forcing
(https://github.com/guandeh17/Self-Forcing).

The model generates a video block by block. Every forward call processes one block of latent
frames: its tokens attend to the cached keys / values of earlier frames plus the block itself,
and the block's own keys / values are written into the cache at the block's position. Calling
the model again at the same position overwrites that entry, which is how a block is first
denoised (noisy K/V) and then committed to history (clean K/V).
"""

import math

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .attention import attention

ROPE_MAX_POSITIONS = 8193  # temporal positions of one session: 8192 latents plus a KV-only sink


def sinusoidal_embedding(dim: int, position: torch.Tensor) -> torch.Tensor:
    """position [N] -> float64 [N, dim] (cosines first, then sines)."""
    half = dim // 2
    position = position.type(torch.float64)
    indices = torch.arange(half, dtype=torch.float64, device=position.device)
    sinusoid = torch.outer(position, torch.pow(10000, -indices.div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def rope_params(max_positions: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Complex128 rotations [max_positions, dim / 2] for one RoPE axis."""
    exponents = torch.arange(0, dim, 2, device="cpu").to(torch.float64).div(dim)
    angles = torch.outer(torch.arange(max_positions, device="cpu"), 1.0 / torch.pow(theta, exponents))
    return torch.polar(torch.ones_like(angles), angles)


@torch.autocast("cuda", enabled=False)
def rope_apply(x: torch.Tensor, grid: tuple[int, int, int], freqs: torch.Tensor, start_frame: int) -> torch.Tensor:
    """Rotate x [B, F*h*w, heads, head_dim] by its 3D position; frames are numbered from `start_frame`.

    The rotation runs in float64 and the result is cast back to the dtype of x.
    """
    f, h, w = grid
    half = x.size(3) // 2
    freqs_t, freqs_h, freqs_w = freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
    rotation = torch.cat([
        freqs_t[start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(1, f * h * w, 1, -1)
    pairs = torch.view_as_complex(x.to(torch.float64).unflatten(3, (half, 2)))
    return torch.view_as_real(pairs * rotation).flatten(3).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight


class LayerNorm(nn.LayerNorm):
    """LayerNorm that returns the input dtype (under autocast the normalisation itself runs in float32)."""

    def __init__(self, dim: int, eps: float, elementwise_affine: bool = False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, local_attn_size: int, sink_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)

    def forward(self, x: torch.Tensor, grid: tuple[int, int, int], freqs: torch.Tensor,
                kv_cache: dict, current_start: int) -> torch.Tensor:
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Keys are cached after RoPE, so every cached token keeps its absolute frame position.
        frame_tokens = grid[1] * grid[2]
        start_frame = current_start // frame_tokens
        q = rope_apply(q, grid, freqs, start_frame).type_as(v)
        k = rope_apply(k, grid, freqs, start_frame).type_as(v)

        # `global_end_index` counts every token written so far; `local_end_index` is the fill level
        # of the cache buffer. They differ once old frames have been evicted.
        current_end = current_start + s
        cache_size = kv_cache["k"].shape[1]
        global_end, local_end = kv_cache["global_end_index"], kv_cache["local_end_index"]
        is_new_block = current_end > global_end
        if is_new_block and local_end + s > cache_size:
            # Buffer full: drop the oldest frames after the sink frames (the first `sink_size`
            # frames of the video are never evicted) and shift the remaining ones left.
            sink = self.sink_size * frame_tokens
            evicted = local_end + s - cache_size
            kept = local_end - evicted - sink
            kv_cache["k"][:, sink:sink + kept] = kv_cache["k"][:, sink + evicted:sink + evicted + kept].clone()
            kv_cache["v"][:, sink:sink + kept] = kv_cache["v"][:, sink + evicted:sink + evicted + kept].clone()
            local_end = local_end + current_end - global_end - evicted
        else:
            # Appending with room to spare, or rewriting the block written last.
            local_end = local_end + current_end - global_end
        kv_cache["k"][:, local_end - s:local_end] = k
        kv_cache["v"][:, local_end - s:local_end] = v
        kv_cache["global_end_index"] = current_end
        kv_cache["local_end_index"] = local_end

        window_start = max(0, local_end - self.local_attn_size * frame_tokens)
        x = attention(q, kv_cache["k"][:, window_start:local_end], kv_cache["v"][:, window_start:local_end])
        return self.o(x.flatten(2))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None, cache: dict) -> torch.Tensor:
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if not cache["is_init"]:
            # The text keys / values do not depend on the latent; compute them once per prompt.
            cache["k"] = self.norm_k(self.k(context)).view(b, -1, n, d)
            cache["v"] = self.v(context).view(b, -1, n, d)
            cache["is_init"] = True
        # Padded text positions are attended like any other token (no key mask), as in training.
        x = attention(q, cache["k"], cache["v"])
        return self.o(x.flatten(2))


class CausalAttentionBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float, local_attn_size: int, sink_size: int):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = CausalSelfAttention(dim, num_heads, eps, local_attn_size, sink_size)
        self.norm3 = LayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor, grid: tuple[int, int, int], freqs: torch.Tensor,
                context: torch.Tensor | None, kv_cache: dict, crossattn_cache: dict,
                current_start: int) -> torch.Tensor:
        """x [B, F*h*w, dim]; e [B, F, 6, dim] per-frame timestep modulation."""
        per_frame = (e.shape[1], x.shape[1] // e.shape[1])
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn(
            (self.norm1(x).unflatten(1, per_frame) * (1 + e[1]) + e[0]).flatten(1, 2),
            grid, freqs, kv_cache, current_start)
        x = x + (y.unflatten(1, per_frame) * e[2]).flatten(1, 2)

        x = x + self.cross_attn(self.norm3(x), context, crossattn_cache)

        y = self.ffn((self.norm2(x).unflatten(1, per_frame) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(1, per_frame) * e[5]).flatten(1, 2)
        return x


class CausalHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float):
        super().__init__()
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """x [B, F*h*w, dim]; e [B, F, 1, dim] -> [B, F, h*w, prod(patch_size) * out_dim]."""
        per_frame = (e.shape[1], x.shape[1] // e.shape[1])
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        return self.head(self.norm(x).unflatten(1, per_frame) * (1 + e[1]) + e[0])


class CausalWanModel(nn.Module):
    """Wan2.2 5B Control DiT with frame-causal self-attention."""

    def __init__(self, *, in_dim: int = 148, dim: int = 3072, ffn_dim: int = 14336, freq_dim: int = 256,
                 text_dim: int = 4096, out_dim: int = 48, num_heads: int = 24, num_layers: int = 30,
                 text_len: int = 512, patch_size: tuple[int, int, int] = (1, 2, 2), eps: float = 1e-6,
                 local_attn_size: int = 21, sink_size: int = 1):
        super().__init__()
        head_dim = dim // num_heads
        if dim % num_heads or head_dim % 2:
            raise ValueError("dim must split into heads of even width")
        if not 0 <= sink_size < local_attn_size:
            raise ValueError("sink_size must be smaller than local_attn_size")
        self.in_dim = in_dim
        self.dim = dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.text_len = text_len
        self.patch_size = tuple(patch_size)
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            CausalAttentionBlock(dim, ffn_dim, num_heads, eps, local_attn_size, sink_size)
            for _ in range(num_layers)
        ])
        self.head = CausalHead(dim, out_dim, self.patch_size, eps)

        # RoPE rotations for the (frame, height, width) axes. Kept as a plain attribute, not a
        # buffer, so that module.to(dtype) cannot change its complex128 precision.
        self.freqs = torch.cat([
            rope_params(ROPE_MAX_POSITIONS, head_dim - 4 * (head_dim // 6)),
            rope_params(ROPE_MAX_POSITIONS, 2 * (head_dim // 6)),
            rope_params(ROPE_MAX_POSITIONS, 2 * (head_dim // 6)),
        ], dim=1)

    def init_kv_cache(self, batch_size: int, frame_tokens: int, device, dtype) -> list[dict]:
        """One dict per layer: k, v [B, local_attn_size*frame_tokens, heads, head_dim], global_end_index, local_end_index."""
        shape = (batch_size, self.local_attn_size * frame_tokens, self.num_heads, self.dim // self.num_heads)
        return [
            {"k": torch.zeros(shape, dtype=dtype, device=device),
             "v": torch.zeros(shape, dtype=dtype, device=device),
             "global_end_index": 0, "local_end_index": 0}
            for _ in range(self.num_layers)
        ]

    def init_crossattn_cache(self, batch_size: int, device, dtype) -> list[dict]:
        """One dict per layer: k, v [B, text_len, heads, head_dim], is_init (bool). Setting is_init=False forces recompute."""
        shape = (batch_size, self.text_len, self.num_heads, self.dim // self.num_heads)
        return [
            {"k": torch.zeros(shape, dtype=dtype, device=device),
             "v": torch.zeros(shape, dtype=dtype, device=device),
             "is_init": False}
            for _ in range(self.num_layers)
        ]

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor, y: torch.Tensor,
                kv_cache: list[dict], crossattn_cache: list[dict], current_start: int) -> torch.Tensor:
        """
        x        [B, 48, F, h, w]  noisy (or clean) latent block
        y        [B, 100, F, h, w] control condition (48 encoded control channels + 52 zeros); concatenated to x on dim 1
        t        [B, F]            timestep per latent frame (0..1000)
        context  [B, 512, 4096]    text embeddings, zero-padded to text_len
        kv_cache / crossattn_cache as returned by init_*; self-attention K/V of this block are written at `current_start`
        current_start  int         token offset of the block = first_latent_frame_index * frame_tokens
        returns  [B, 48, F, h, w]  flow prediction
        """
        weight = self.patch_embedding.weight
        if self.freqs.device != weight.device:
            self.freqs = self.freqs.to(weight.device)
        if context.shape[1] != self.text_len:
            raise ValueError(f"context must be padded to {self.text_len} tokens, got {context.shape[1]}")

        # Half-precision weights run under autocast: linear layers stay in the weight dtype while
        # the normalisation layers are computed in float32.
        half_precision = weight.dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(weight.device.type, dtype=weight.dtype, enabled=half_precision):
            x = self.patch_embedding(torch.cat([x, y], dim=1))
            b, _, f, h, w = x.shape
            grid = (f, h, w)
            x = x.flatten(2).transpose(1, 2)

            e = self.time_embedding(sinusoidal_embedding(self.freq_dim, t.flatten()).type_as(x))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)

            # Once every layer holds the text keys / values, the text embedding is not needed.
            if all(cache["is_init"] for cache in crossattn_cache):
                context = None
            else:
                context = self.text_embedding(context)

            for block, block_kv, block_cross in zip(self.blocks, kv_cache, crossattn_cache):
                x = block(x, e0, grid, self.freqs, context, block_kv, block_cross, current_start)

            x = self.head(x, e.unflatten(0, t.shape).unsqueeze(2))

        # Unpatchify: [B, F, h*w, pt*ph*pw*C] -> [B, C, F*pt, h*ph, w*pw].
        pt, ph, pw = self.patch_size
        x = x.view(b, f, h, w, pt, ph, pw, self.out_dim)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(b, self.out_dim, f * pt, h * ph, w * pw)


# Parameters of the base architecture that the causal inference path never evaluates.
_UNUSED_KEY_PREFIXES = ("ref_conv.",)
_CHECKPOINT_KEY_PREFIXES = ("pipe.dit.", "model.")


def load_causal_wan(checkpoint: str, *, local_attn_size: int = 21, sink_size: int = 1,
                    device="cuda", dtype=torch.bfloat16) -> CausalWanModel:
    """Build the model and load the student safetensors (eval mode, requires_grad False)."""
    state = load_file(checkpoint, device="cpu")
    for prefix in _CHECKPOINT_KEY_PREFIXES:
        if any(key.startswith(prefix) for key in state):
            state = {(key[len(prefix):] if key.startswith(prefix) else key): value for key, value in state.items()}
    state = {key: value for key, value in state.items() if not key.startswith(_UNUSED_KEY_PREFIXES)}

    # Parameters are created without storage and replaced by the checkpoint tensors (assign=True),
    # which skips the random initialisation of ~5B parameters.
    with torch.device("meta"):
        model = CausalWanModel(local_attn_size=local_attn_size, sink_size=sink_size)
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device=device, dtype=dtype)
    return model.eval().requires_grad_(False)
