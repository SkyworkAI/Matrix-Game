"""FP8 block linears in layouts that let cuBLAS pick its fast sm_120 kernels.

For M = 405 tokens cuBLAS reaches ~170-195 TFLOPS; with rows padded to 16, no bias in the GEMM
(added by a Triton epilogue) and the feed-forward computed transposed it reaches ~300. Weights,
scales and rounding points are those of ``FP8Linear``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn

from ..rtx5090.fp8_linear import _MAX_FEATURES, _MIN_SCALE, FP8Linear, _row_absmax

ROW_MULTIPLE = 16


def padded_rows(rows: int) -> int:
    return -(-rows // ROW_MULTIPLE) * ROW_MULTIPLE


@triton.jit
def _scale_quantize_rows_padded(x_ptr, y_ptr, max_ptr, scale_ptr, rows, cols, stride_m, stride_k,
                                BLOCK: tl.constexpr, ROWS: tl.constexpr, MIN_SCALE: tl.constexpr):
    # `_scale_quantize_rows` over padded rows; rows >= `rows` are stored as zeros.
    row = tl.program_id(0)
    index = tl.arange(0, ROWS)
    values = tl.load(max_ptr + index, index < rows, 0.0)
    scale = tl.maximum(tl.max(values, 0) / 448.0, MIN_SCALE)
    if row == 0:
        tl.store(scale_ptr, scale)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * stride_m + offs * stride_k, mask=mask & (row < rows), other=0.0)
    q = tl.minimum(tl.maximum(x.to(tl.float32) / scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(y_ptr + row * cols + offs, q, mask=mask)


def quantize_activation_padded(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``fp8_linear.quantize_activation`` with the result padded to a multiple of 16 rows (zeros).

    Returns (e4m3 ``[padded_rows(M), K]``, FP32 scale ``[1]``); the scale covers the real rows only.
    """
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("activation quantizer expects a CUDA BF16 [M, K] tensor")
    rows, cols = x.shape
    if rows < 1 or not 0 < cols <= _MAX_FEATURES:
        raise ValueError(f"activation quantizer needs M > 0 and 0 < K <= {_MAX_FEATURES}")
    block = max(16, triton.next_power_of_2(cols))
    padded = padded_rows(rows)
    row_max = torch.empty((rows,), device=x.device, dtype=torch.float32)
    scale = torch.empty((1,), device=x.device, dtype=torch.float32)
    y = torch.empty((padded, cols), device=x.device, dtype=torch.float8_e4m3fn)
    with torch.cuda.device(x.device):
        _row_absmax[(rows,)](x, row_max, cols, x.stride(0), x.stride(1), BLOCK=block)
        _scale_quantize_rows_padded[(padded,)](x, y, row_max, scale, rows, cols, x.stride(0), x.stride(1),
                                               BLOCK=block, ROWS=triton.next_power_of_2(rows), MIN_SCALE=_MIN_SCALE)
    return y, scale


@triton.jit
def _bias_rows_kernel(src_ptr, bias_ptr, dst_ptr, M, N, src_stride_s, src_stride_m, dst_stride_s,
                      BM: tl.constexpr, BN: tl.constexpr):
    # dst[s] = bf16(src[s] + bias[s]) for S stacked FP32 products [padded M, N].
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    s = tl.program_id(2)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    acc = tl.load(src_ptr + s * src_stride_s + rm[:, None] * src_stride_m + rn[None, :], mask=mask, other=0.0)
    y = acc + tl.load(bias_ptr + s * N + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    tl.store(dst_ptr + s * dst_stride_s + rm[:, None] * N + rn[None, :], y.to(tl.bfloat16), mask=mask)


def _padded_gemms(linears, x2: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """BF16 [M, K] through the given ``FP8Linear``s (same input, same out_features) -> BF16 [S, M, N];
    `bias` [S, N] holds their biases.

    The GEMMs run without a bias (with one, cuBLAS falls back to a ~1.5x slower sm89 kernel) into FP32
    and a single Triton kernel adds every bias and rounds to BF16 once, as the GEMM epilogue would.
    """
    qx, x_scale = quantize_activation_padded(x2)
    m, n = x2.shape[0], linears[0].out_features
    acc = torch.empty((len(linears), qx.shape[0], n), device=x2.device, dtype=torch.float32)
    for index, linear in enumerate(linears):
        torch._scaled_mm(qx, linear.weight, x_scale, linear.weight_scale, out_dtype=torch.float32, out=acc[index])
    out = torch.empty((len(linears), m, n), device=x2.device, dtype=torch.bfloat16)
    _bias_rows_kernel[(triton.cdiv(m, 64), triton.cdiv(n, 128), len(linears))](
        acc, bias, out, m, n, acc.stride(0), acc.stride(1), out.stride(0), BM=64, BN=128, num_warps=4)
    return out


class PaddedFP8Linear(FP8Linear):
    """``FP8Linear`` through ``_padded_gemms`` (same weights and scales, faster cuBLAS kernel)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        if self.bias is None:
            raise ValueError("PaddedFP8Linear needs a bias")
        return _padded_gemms((self,), x.to(torch.bfloat16).reshape(-1, self.in_features), self.bias)[0].reshape(
            *x.shape[:-1], self.out_features)

    @classmethod
    def from_fp8(cls, linear: FP8Linear) -> "PaddedFP8Linear":
        linear.__class__ = cls  # same storage and buffers; only the forward differs
        return linear


def project_qkv(attn: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q / k / v of a self-attention from one quantisation and one bias epilogue (the fused block's
    ``project_qkv``). Needs the stacked biases ``attn.qkv_bias`` set by ``quantize_blocks``."""
    linears = (attn.q, attn.k, attn.v)
    for linear in linears:
        linear._check(x)
    out = _padded_gemms(linears, x.to(torch.bfloat16).reshape(-1, x.shape[-1]), attn.qkv_bias)
    return tuple(part.reshape(*x.shape[:-1], part.shape[-1]) for part in out)


@triton.jit
def _bias_transpose_kernel(src_ptr, bias_ptr, dst_ptr, M, N, ld_src,
                           GELU: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # dst [M, N] = bf16(src^T + bias), optionally followed by the tanh GELU; src is [N, padded M] FP32.
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    acc = tl.load(src_ptr + rn[None, :] * ld_src + rm[:, None], mask=mask, other=0.0)
    y = (acc + tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]).to(tl.bfloat16)
    if GELU:
        v = y.to(tl.float32)
        inner = 0.7978845608028654 * (v + 0.044715 * v * v * v)
        y = (0.5 * v * (1.0 + tl.extra.cuda.libdevice.tanh(inner))).to(tl.bfloat16)
    tl.store(dst_ptr + rm[:, None] * N + rn[None, :], y, mask=mask)


def _transposed_linear(linear: FP8Linear, x2: torch.Tensor, gelu: bool) -> torch.Tensor:
    """x2 [M, K] BF16 -> [M, N] BF16 through ``W @ x^T`` (FP32), then transpose + bias (+ GELU)."""
    qx, x_scale = quantize_activation_padded(x2)
    # weight.t() is the row-major [N, K] matrix.
    acc = torch._scaled_mm(linear.weight.t(), qx.t(), linear.weight_scale, x_scale, out_dtype=torch.float32)
    return _finish(acc, linear.bias, x2.shape[0], gelu)


def _finish(acc: torch.Tensor, bias: torch.Tensor, m: int, gelu: bool) -> torch.Tensor:
    """Transposed FP32 product [N, padded M] -> BF16 [M, N] with bias (and GELU)."""
    n = acc.shape[0]
    out = torch.empty((m, n), device=acc.device, dtype=torch.bfloat16)
    _bias_transpose_kernel[(triton.cdiv(m, 64), triton.cdiv(n, 64))](
        acc, bias, out, m, n, acc.stride(0), GELU=gelu, BM=64, BN=64, num_warps=4)
    return out


class TransposedFFN(nn.Module):
    """The block feed-forward (FP8 linear, tanh GELU, FP8 linear) with both GEMMs transposed."""

    def __init__(self, ffn: nn.Sequential):
        super().__init__()
        up, act, down = ffn
        if not (isinstance(up, FP8Linear) and isinstance(down, FP8Linear)) or getattr(act, "approximate", None) != "tanh":
            raise ValueError("expected FP8Linear, GELU(tanh), FP8Linear")
        if up.bias is None or down.bias is None:
            raise ValueError("the feed-forward linears need biases")
        self.up, self.down = up, down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.up._check(x)
        hidden = _transposed_linear(self.up, x.to(torch.bfloat16).reshape(-1, x.shape[-1]), gelu=True)
        return _transposed_linear(self.down, hidden, gelu=False).reshape(*x.shape[:-1], self.down.out_features)


def quantize_blocks(model: nn.Module, device) -> None:
    """Quantise the block linears of a BF16 ``CausalWanModel`` (on the CPU) to FP8 in this runtime's
    layouts (``PaddedFP8Linear``, ``TransposedFFN``), moving each block to `device` when done."""
    for block in model.blocks:
        for attention in (block.self_attn, block.cross_attn):
            for name in ("q", "k", "v", "o"):
                setattr(attention, name, PaddedFP8Linear.from_fp8(FP8Linear(getattr(attention, name)).eval()))
        attn = block.self_attn
        attn.register_buffer("qkv_bias", torch.stack([attn.q.bias, attn.k.bias, attn.v.bias]).detach())
        block.ffn = TransposedFFN(nn.Sequential(FP8Linear(block.ffn[0]), block.ffn[1], FP8Linear(block.ffn[2])))
        block.to(device)
