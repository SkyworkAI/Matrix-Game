"""Triton kernels that fuse the glue between the GEMMs and attention calls of a causal DiT block.

A block is five linear layers and two attention calls connected by long chains of tiny element-wise
and row-reduction operations (LayerNorm, timestep modulation, gated residuals, RMSNorm, RoPE). For
one latent frame every such operation moves a few MB, so its cost is the kernel launch, not the
memory traffic. Each kernel below replaces one whole chain with a single launch:

* `ln_modulate`            LayerNorm(x) * (1 + scale) + shift, where shift / scale are rows of
                           `modulation + e`; optionally adds a residual to x first.
* `gate_ln_affine`         x + y * gate, followed by the affine LayerNorm before cross-attention.
* `gate_residual`          x + y * gate.
* `rms_norm`               RMSNorm of the cross-attention queries.
* `qk_rms_rope`            RMSNorm + RoPE of the self-attention queries and keys in one launch; the
                           keys are written straight into their slot of the KV cache.

Rounding follows the unfused BF16 block: every intermediate that the chain of separate operations
would materialise in BF16 is rounded to BF16 here too, and all reductions accumulate in FP32. Only
the summation order inside a row reduction differs. RoPE is rotated in FP32.

All kernels are launched on the current stream without any host synchronisation, so a block built
from them can be captured in a CUDA graph. Inputs are checked on the host and rejected with an
error; nothing falls back to an unfused path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MODULATION_ROWS = 6  # shift / scale / gate rows of a block's timestep modulation


# ----------------------------------------------------------------------------------------------
# Shared kernel pieces
# ----------------------------------------------------------------------------------------------

@triton.jit
def _bf16(v):
    # Round an FP32 value to BF16 precision (round-to-nearest-even) and return it as FP32, the way
    # a separate BF16 operation would materialise its result. Done on the bit pattern because the
    # compiler elides a `.to(bfloat16).to(float32)` round trip and with it the rounding.
    u = v.to(tl.int32, bitcast=True)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & -65536
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _modulation(mod_ptr, e_ptr, k, cols, m, C):
    # Row k of `modulation + e`, rounded to BF16 like the separate add.
    mv = tl.load(mod_ptr + k * C + cols, mask=m, other=0.0).to(tl.float32)
    ev = tl.load(e_ptr + k * C + cols, mask=m, other=0.0).to(tl.float32)
    return _bf16(mv + ev)


@triton.jit
def _row_stats(x, m, C, eps):
    # LayerNorm statistics of one row: centred values and 1 / sqrt(var + eps).
    mean = tl.sum(x, axis=0) / C
    xc = tl.where(m, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / C
    return xc, tl.rsqrt(var + eps)


def _block_size(n: int) -> int:
    return triton.next_power_of_2(n)


def _warps(block: int) -> int:
    return 1 if block <= 256 else 4 if block <= 1024 else 8


def _check_rows(name: str, x: torch.Tensor) -> None:
    if x.dtype != torch.bfloat16 or x.ndim != 3 or not x.is_contiguous() or not x.is_cuda:
        raise ValueError(f"{name} must be a contiguous BF16 CUDA tensor [B, S, C], got "
                         f"{tuple(x.shape)} {x.dtype} on {x.device}")


def _check_vector(name: str, v: torch.Tensor, channels: int) -> None:
    if v.dtype != torch.bfloat16 or v.shape != (channels,) or not v.is_contiguous():
        raise ValueError(f"{name} must be a contiguous BF16 vector of {channels} elements, got "
                         f"{tuple(v.shape)} {v.dtype}")


def _modulation_strides(x: torch.Tensor, e: torch.Tensor, modulation: torch.Tensor) -> tuple[int, int, int]:
    """Validate e [B, F, 6, C] / modulation [1, 6, C]; returns (tokens per frame, e strides over B and F).

    e may be a view (for example expanded over the batch), as long as its last two dimensions are
    laid out densely; the kernels address it through the two returned strides.
    """
    b, s, c = x.shape
    if (e.dtype != torch.bfloat16 or e.ndim != 4 or e.shape[0] != b or e.shape[2:] != (MODULATION_ROWS, c)
            or e.stride(3) != 1 or e.stride(2) != c):
        raise ValueError(f"e must be BF16 [B, F, {MODULATION_ROWS}, C] with dense last two dimensions, got "
                         f"{tuple(e.shape)} {e.dtype} strides {tuple(e.stride())}")
    if (modulation.dtype != torch.bfloat16 or modulation.shape != (1, MODULATION_ROWS, c)
            or not modulation.is_contiguous()):
        raise ValueError(f"modulation must be contiguous BF16 [1, {MODULATION_ROWS}, C], got "
                         f"{tuple(modulation.shape)} {modulation.dtype}")
    if s % e.shape[1]:
        raise ValueError(f"{s} tokens do not divide into {e.shape[1]} frames")
    return s // e.shape[1], e.stride(0), e.stride(1)


# ----------------------------------------------------------------------------------------------
# LayerNorm + timestep modulation (before self-attention and before the feed-forward)
# ----------------------------------------------------------------------------------------------

@triton.jit
def _ln_modulate_kernel(
    x_ptr, y_ptr, e_ptr, mod_ptr, z_ptr, xo_ptr,
    seq_len, frame_tokens, C, e_sb, e_sf, eps,
    SHIFT_K: tl.constexpr, SCALE_K: tl.constexpr, ADD_RES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    f = (row % seq_len) // frame_tokens
    cols = tl.arange(0, BLOCK)
    m = cols < C
    x = tl.load(x_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
    if ADD_RES:
        y = tl.load(y_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
        x = _bf16(x + y)
        tl.store(xo_ptr + row * C + cols, x.to(tl.bfloat16), mask=m)
    xc, rstd = _row_stats(x, m, C, eps)
    n = _bf16(xc * rstd)
    e_row = e_ptr + b * e_sb + f * e_sf
    shift = _modulation(mod_ptr, e_row, SHIFT_K, cols, m, C)
    scale = _bf16(1.0 + _modulation(mod_ptr, e_row, SCALE_K, cols, m, C))
    z = _bf16(n * scale) + shift
    tl.store(z_ptr + row * C + cols, z.to(tl.bfloat16), mask=m)


def ln_modulate(x: torch.Tensor, e: torch.Tensor, modulation: torch.Tensor, eps: float,
                shift_k: int, scale_k: int, residual: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """z = LayerNorm(x') * (1 + (modulation + e)[scale_k]) + (modulation + e)[shift_k]; returns (z, x').

    x' is `x + residual` (a new BF16 tensor) when `residual` is given, otherwise `x` itself.
    x, residual [B, S, C]; e [B, F, 6, C] (one modulation per frame); modulation [1, 6, C].
    """
    _check_rows("x", x)
    b, s, c = x.shape
    frame_tokens, e_sb, e_sf = _modulation_strides(x, e, modulation)
    add_residual = residual is not None
    if add_residual:
        _check_rows("residual", residual)
        if residual.shape != x.shape:
            raise ValueError(f"residual {tuple(residual.shape)} does not match x {tuple(x.shape)}")
    z = torch.empty_like(x)
    x_new = torch.empty_like(x) if add_residual else x
    block = _block_size(c)
    _ln_modulate_kernel[(b * s,)](
        x, residual if add_residual else x, e, modulation, z, x_new,
        s, frame_tokens, c, e_sb, e_sf, eps,
        SHIFT_K=shift_k, SCALE_K=scale_k, ADD_RES=add_residual,
        BLOCK=block, num_warps=_warps(block),
    )
    return z, x_new


# ----------------------------------------------------------------------------------------------
# Gated residuals (after self-attention, with the following affine LayerNorm; after the feed-forward)
# ----------------------------------------------------------------------------------------------

@triton.jit
def _gate_ln_affine_kernel(
    x_ptr, y_ptr, e_ptr, mod_ptr, w_ptr, bias_ptr, z_ptr, xo_ptr,
    seq_len, frame_tokens, C, e_sb, e_sf, eps,
    GATE_K: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    f = (row % seq_len) // frame_tokens
    cols = tl.arange(0, BLOCK)
    m = cols < C
    x = tl.load(x_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
    gate = _modulation(mod_ptr, e_ptr + b * e_sb + f * e_sf, GATE_K, cols, m, C)
    x = _bf16(x + _bf16(y * gate))
    tl.store(xo_ptr + row * C + cols, x.to(tl.bfloat16), mask=m)
    xc, rstd = _row_stats(x, m, C, eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=m, other=0.0).to(tl.float32)
    # The affine LayerNorm computes normalisation and affine in FP32 and rounds once.
    tl.store(z_ptr + row * C + cols, (xc * rstd * w + bias).to(tl.bfloat16), mask=m)


def gate_ln_affine(x: torch.Tensor, y: torch.Tensor, e: torch.Tensor, modulation: torch.Tensor,
                   gate_k: int, weight: torch.Tensor, bias: torch.Tensor, eps: float
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """x' = x + y * (modulation + e)[gate_k] and z = LayerNorm(x') * weight + bias; returns (z, x').

    x, y [B, S, C]; e [B, F, 6, C]; modulation [1, 6, C]; weight, bias [C]. Both results are new tensors.
    """
    _check_rows("x", x)
    _check_rows("y", y)
    if y.shape != x.shape:
        raise ValueError(f"y {tuple(y.shape)} does not match x {tuple(x.shape)}")
    b, s, c = x.shape
    frame_tokens, e_sb, e_sf = _modulation_strides(x, e, modulation)
    _check_vector("weight", weight, c)
    _check_vector("bias", bias, c)
    z = torch.empty_like(x)
    x_new = torch.empty_like(x)
    block = _block_size(c)
    _gate_ln_affine_kernel[(b * s,)](
        x, y, e, modulation, weight, bias, z, x_new,
        s, frame_tokens, c, e_sb, e_sf, eps,
        GATE_K=gate_k, BLOCK=block, num_warps=_warps(block),
    )
    return z, x_new


@triton.jit
def _gate_residual_kernel(
    x_ptr, y_ptr, e_ptr, mod_ptr, xo_ptr,
    seq_len, frame_tokens, C, e_sb, e_sf,
    GATE_K: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    f = (row % seq_len) // frame_tokens
    cols = tl.arange(0, BLOCK)
    m = cols < C
    x = tl.load(x_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
    gate = _modulation(mod_ptr, e_ptr + b * e_sb + f * e_sf, GATE_K, cols, m, C)
    tl.store(xo_ptr + row * C + cols, _bf16(x + _bf16(y * gate)).to(tl.bfloat16), mask=m)


def gate_residual(x: torch.Tensor, y: torch.Tensor, e: torch.Tensor, modulation: torch.Tensor,
                  gate_k: int) -> torch.Tensor:
    """x + y * (modulation + e)[gate_k], with the product and the sum each rounded to BF16.

    x, y [B, S, C]; e [B, F, 6, C]; modulation [1, 6, C]. Returns a new [B, S, C] tensor.
    """
    _check_rows("x", x)
    _check_rows("y", y)
    if y.shape != x.shape:
        raise ValueError(f"y {tuple(y.shape)} does not match x {tuple(x.shape)}")
    b, s, c = x.shape
    frame_tokens, e_sb, e_sf = _modulation_strides(x, e, modulation)
    out = torch.empty_like(x)
    block = _block_size(c)
    _gate_residual_kernel[(b * s,)](
        x, y, e, modulation, out,
        s, frame_tokens, c, e_sb, e_sf,
        GATE_K=gate_k, BLOCK=block, num_warps=_warps(block),
    )
    return out


# ----------------------------------------------------------------------------------------------
# RMSNorm (cross-attention queries)
# ----------------------------------------------------------------------------------------------

@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, C, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < C
    x = tl.load(x_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) * (1.0 / C)
    n = _bf16(x * tl.rsqrt(ms + eps))
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * C + cols, (n * w).to(tl.bfloat16), mask=m)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension of x [B, S, C] with weight [C]; returns a new [B, S, C] tensor."""
    _check_rows("x", x)
    b, s, c = x.shape
    _check_vector("weight", weight, c)
    out = torch.empty_like(x)
    block = _block_size(c)
    _rms_norm_kernel[(b * s,)](x, weight, out, c, eps, BLOCK=block, num_warps=_warps(block),
                               enable_fp_fusion=False)
    return out


# ----------------------------------------------------------------------------------------------
# RMSNorm + RoPE of the self-attention queries and keys, keys stored in the KV cache
# ----------------------------------------------------------------------------------------------

def rope_rotations(freqs: torch.Tensor, start_frame: int, f: int, h: int, w: int) -> torch.Tensor:
    """FP32 rotation table [f*h*w, head_dim/2, 2] of (cos, sin) for a block of f frames on an h x w grid.

    `freqs` is the model's complex table [positions, head_dim/2] whose columns are split over the
    time / height / width axes; frames are numbered from `start_frame` (absolute positions).
    """
    if freqs.ndim != 2 or not torch.is_complex(freqs):
        raise ValueError("freqs must be a complex [positions, head_dim/2] table")
    half = freqs.shape[1]
    freqs_t, freqs_h, freqs_w = freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
    if start_frame < 0 or start_frame + f > freqs.shape[0] or max(h, w) > freqs.shape[0]:
        raise ValueError(f"RoPE grid (frames {start_frame}..{start_frame + f}, {h}x{w}) exceeds the "
                         f"{freqs.shape[0]} positions of the frequency table")
    rotation = torch.cat([
        freqs_t[start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, half)
    return torch.view_as_real(rotation).to(torch.float32).contiguous()


@triton.jit
def _rms_rope_row(x_ptr, w_ptr, rot_ptr, out_ptr, token, row, C, PAIRS, HALF, eps,
                  BLOCK: tl.constexpr):
    # One [heads * head_dim] row; adjacent elements form a complex pair and the rotation table
    # repeats every HALF pairs (one head).
    p = tl.arange(0, BLOCK)
    m = p < PAIRS
    xr = tl.load(x_ptr + row * C + 2 * p, mask=m, other=0.0).to(tl.float32)
    xi = tl.load(x_ptr + row * C + 2 * p + 1, mask=m, other=0.0).to(tl.float32)
    ms = tl.sum(xr * xr + xi * xi, axis=0) * (1.0 / C)
    r = tl.rsqrt(ms + eps)
    wr = tl.load(w_ptr + 2 * p, mask=m, other=0.0).to(tl.float32)
    wi = tl.load(w_ptr + 2 * p + 1, mask=m, other=0.0).to(tl.float32)
    # RMSNorm: bf16(x * rsqrt) * weight, each rounded to BF16.
    nr = _bf16(_bf16(xr * r) * wr)
    ni = _bf16(_bf16(xi * r) * wi)
    rot = rot_ptr + (token * HALF + p % HALF) * 2
    cos = tl.load(rot, mask=m, other=0.0)
    sin = tl.load(rot + 1, mask=m, other=0.0)
    out_r = nr * cos - ni * sin
    out_i = nr * sin + ni * cos
    tl.store(out_ptr + 2 * p, out_r.to(tl.bfloat16), mask=m)
    tl.store(out_ptr + 2 * p + 1, out_i.to(tl.bfloat16), mask=m)


@triton.jit
def _qk_rms_rope_kernel(
    q_ptr, k_ptr, wq_ptr, wk_ptr, rot_ptr, qo_ptr, ko_ptr,
    seq_len, C, PAIRS, HALF, cache_stride_b, cache_row0, eps,
    BLOCK: tl.constexpr,
):
    # Grid [rows, 2]: the second axis selects the query row or the key row of the same token.
    row = tl.program_id(0)
    which = tl.program_id(1)
    b = row // seq_len
    token = row % seq_len
    if which == 0:
        _rms_rope_row(q_ptr, wq_ptr, rot_ptr, qo_ptr + row * C, token, row, C, PAIRS,
                      HALF, eps, BLOCK)
    else:
        _rms_rope_row(k_ptr, wk_ptr, rot_ptr,
                      ko_ptr + b * cache_stride_b + (cache_row0 + token) * C,
                      token, row, C, PAIRS, HALF, eps, BLOCK)


def qk_rms_rope(q: torch.Tensor, k: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor,
                rotations: torch.Tensor, cache_k: torch.Tensor, start: int, num_heads: int,
                eps: float) -> torch.Tensor:
    """RMSNorm + RoPE of the projected queries and keys in one launch; returns the queries.

    q, k [B, S, C] are the outputs of the q / k projections, weight_q / weight_k [C] the RMSNorm
    weights and `rotations` the FP32 table of `rope_rotations` for these S tokens. The result is a
    new [B, S, num_heads, C / num_heads] tensor; the keys are not returned but written into
    `cache_k[:, start:start + S]` (cache_k is the contiguous [B, capacity, num_heads, head_dim] cache).
    """
    _check_rows("q", q)
    _check_rows("k", k)
    if k.shape != q.shape:
        raise ValueError(f"k {tuple(k.shape)} does not match q {tuple(q.shape)}")
    b, s, c = q.shape
    if c % (2 * num_heads):
        raise ValueError(f"{c} channels do not split into {num_heads} heads of complex pairs")
    head_dim = c // num_heads
    _check_vector("weight_q", weight_q, c)
    _check_vector("weight_k", weight_k, c)
    if (rotations.dtype != torch.float32 or rotations.shape != (s, head_dim // 2, 2)
            or not rotations.is_contiguous()):
        raise ValueError(f"rotations must be contiguous FP32 [{s}, {head_dim // 2}, 2] (see rope_rotations), "
                         f"got {tuple(rotations.shape)} {rotations.dtype}")
    if (cache_k.dtype != torch.bfloat16 or cache_k.ndim != 4 or not cache_k.is_contiguous()
            or cache_k.shape[0] != b or cache_k.shape[2:] != (num_heads, head_dim)
            or start < 0 or start + s > cache_k.shape[1]):
        raise ValueError(f"cache_k must be contiguous BF16 [{b}, capacity, {num_heads}, {head_dim}] with room for "
                         f"tokens {start}..{start + s}, got {tuple(cache_k.shape)} {cache_k.dtype}")
    q_out = torch.empty(b, s, num_heads, head_dim, dtype=q.dtype, device=q.device)
    block = _block_size(c // 2)
    _qk_rms_rope_kernel[(b * s, 2)](
        q, k, weight_q, weight_k, rotations, q_out, cache_k,
        s, c, c // 2, head_dim // 2, cache_k.stride(0), start, eps,
        BLOCK=block, num_warps=_warps(block),
        # Keep the rotation as separate multiplies and adds (no FMA contraction).
        enable_fp_fusion=False,
    )
    return q_out


# ----------------------------------------------------------------------------------------------
# Merge of attention computed over disjoint key ranges
# ----------------------------------------------------------------------------------------------

__all__ = [
    "gate_ln_affine", "gate_residual", "ln_modulate", "qk_rms_rope",
    "rms_norm", "rope_rotations",
]
