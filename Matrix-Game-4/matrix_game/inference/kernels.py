# Copyright 2026 Skywork AI and contributors.
# Apache-2.0; see the repository LICENSE. Only norm/modulation/residual kernels.
"""Triton kernels that fuse the glue between the GEMMs and attention calls of a causal DiT block.

A block is five linear layers and two attention calls connected by long chains of tiny element-wise
and row-reduction operations (LayerNorm, timestep modulation, gated residuals, RMSNorm, RoPE). For
one latent frame every such operation moves a few MB, so its cost is the kernel launch, not the
memory traffic. Each kernel below replaces one whole chain with a single launch:

* `ln_modulate`            LayerNorm(x) * (1 + scale) + shift, where shift / scale are rows of
                           `modulation + e`; optionally adds a residual to x first.
* `gate_ln_affine`         x + y * gate, followed by the affine LayerNorm before cross-attention.
* `gate_residual`          x + y * gate.

Rounding follows the unfused BF16 block: every intermediate that the chain of separate operations
would materialise in BF16 is rounded to BF16 here too, and all reductions accumulate in FP32. Only
the summation order inside a row reduction differs. RoPE is not changed here.

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



