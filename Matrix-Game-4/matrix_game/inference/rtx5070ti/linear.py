"""INT8 (W8A8) block linears; MXFP8 for the cross-attention query, whose input has outlier channels.

Block linears of one forward at M = 405: INT8 ~16.5 ms, RTX 5080 FP8 ~21 ms, RTX 5090 FP8 ~23 ms.
q / k / v are one GEMM; the FFN's GELU runs in the INT8 GEMM epilogue.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice

from ..rtx5090.fp8_linear import FP8Linear, project_shared_input
from ..rtx5090.kernels import _bf16

_MAX_FEATURES = 32768  # one Triton program quantizes a whole row


@triton.jit
def _quantize_rows_kernel(x_ptr, y_ptr, scale_ptr, cols, stride_m, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1e-8)
    q = x / scale
    q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5))
    tl.store(y_ptr + row * cols + offs, q.to(tl.int8), mask=mask)
    tl.store(scale_ptr + row, scale)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``[M, K]`` (unit column stride) -> (INT8 ``[M, K]``, FP32 per-row scale ``[M]``)."""
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError("activation quantizer expects a CUDA BF16 [M, K] tensor with unit column stride")
    rows, cols = x.shape
    if rows < 1 or not 0 < cols <= _MAX_FEATURES:
        raise ValueError(f"activation quantizer needs M > 0 and 0 < K <= {_MAX_FEATURES}")
    y = torch.empty((rows, cols), device=x.device, dtype=torch.int8)
    scale = torch.empty((rows,), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(cols)
    with torch.cuda.device(x.device):
        _quantize_rows_kernel[(rows,)](x, y, scale, cols, x.stride(0), BLOCK=block,
                                       num_warps=8 if block >= 8192 else 4)
    return y, scale


def _int8_tile(n: int, k: int) -> dict:
    """GEMM tile (BM, BN, BK, warps, stages) by shape, the fastest of a sweep on the RTX 5070 Ti at
    M = 405. Fixed rather than autotuned: the epilogue's floating-point code differs between tiles,
    so a timing-based choice could change results from one process to the next."""
    if k > 4096:                 # ffn.2
        bm, bn, bk, warps, stages = 64, 128, 64, 4, 3
    elif n >= 8192:              # fused q / k / v, ffn.0
        bm, bn, bk, warps, stages = 64, 256, 64, 4, 3
    else:                        # 3072 x 3072 projections
        bm, bn, bk, warps, stages = 64, 128, 64, 4, 2
    return dict(BM=bm, BN=bn, BK=bk, GROUP=4, num_warps=warps, num_stages=stages)


@triton.jit
def _int8_gemm_kernel(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, bias_ptr, M, N, K, SPLIT_N, GELU: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    # a [M, K] and b [N, K] are row-major INT8; tiles are visited in groups of GROUP row tiles so
    # that consecutive programs share weight tiles in L2.
    pid = tl.program_id(0)
    tiles_m, tiles_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    width = GROUP * tiles_n
    first = (pid // width) * GROUP
    size = min(tiles_m - first, GROUP)
    pm = first + (pid % width) % size
    pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = a_ptr + (rm % M)[:, None] * K + rk[None, :]
    b = b_ptr + (rn % N)[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BK)):
        x = tl.load(a, mask=rk[None, :] < K - k * BK, other=0)
        y = tl.load(b, mask=rk[:, None] < K - k * BK, other=0)
        acc = tl.dot(x, y, acc, out_dtype=tl.int32)
        a += BK
        b += BK
    sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
    sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
    out = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    out += tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    if GELU:
        # tanh-GELU of the BF16-rounded product, as `nn.GELU("tanh")` applied to the linear's output.
        out = _bf16(out)
        out = 0.5 * out * (1.0 + libdevice.tanh(0.7978845608028654 * (out + 0.044715 * out * out * out)))
    # Column n goes to output part n // SPLIT_N, each a contiguous [M, SPLIT_N] matrix.
    c = c_ptr + (rn // SPLIT_N)[None, :] * (M * SPLIT_N) + rm[:, None] * SPLIT_N + (rn % SPLIT_N)[None, :]
    tl.store(c, out.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def int8_gemm(x: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
              bias: torch.Tensor, split: int, gelu: bool = False) -> torch.Tensor:
    """x INT8 [M, K] with per-row scales, weight INT8 [N, K] with per-row scales, bias BF16 [N]
    -> BF16 [N // split, M, split]: x @ weight.T + bias (then tanh-GELU with `gelu`), columns split
    into contiguous parts."""
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((n // split, m, split), device=x.device, dtype=torch.bfloat16)
    tile = _int8_tile(n, k)
    with torch.cuda.device(x.device):
        _int8_gemm_kernel[(triton.cdiv(m, tile["BM"]) * triton.cdiv(n, tile["BN"]),)](
            x, weight, out, x_scale, weight_scale, bias, m, n, k, split, GELU=gelu, **tile)
    return out


class Int8Linear(nn.Module):
    """One or more frozen linears that read the same input, as one INT8 GEMM.

    Move this module between devices only: a dtype cast would destroy the INT8 storage.
    """

    def __init__(self, linears: list[nn.Linear], gelu: bool = False):
        super().__init__()
        self.in_features = linears[0].in_features
        self.out_features = linears[0].out_features
        self.gelu = gelu  # apply tanh-GELU to the output in the GEMM epilogue
        if any(linear.in_features != self.in_features or linear.out_features != self.out_features
               or linear.bias is None for linear in linears):
            raise ValueError("fused linears need the same shape and a bias")
        if self.in_features % 16 or self.out_features % 128:
            raise ValueError("Int8Linear requires in_features % 16 == 0 and out_features % 128 == 0")
        self.parts = len(linears)
        weight = torch.cat([linear.weight.detach() for linear in linears]).float()
        scale = (weight.abs().amax(dim=1) / 127).clamp(min=1e-8)
        self.register_buffer("weight", torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8))
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", torch.cat([linear.bias.detach() for linear in linears]).to(torch.bfloat16))

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, parts={self.parts}"
                + (", gelu" if self.gelu else ""))

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """x [..., K] -> one contiguous [..., out_features] tensor per fused linear."""
        if self.weight.dtype != torch.int8 or x.device != self.weight.device or x.shape[-1] != self.in_features:
            raise RuntimeError("Int8Linear input must match the weight's device and in_features")
        qx, x_scale = quantize_rows(x.to(torch.bfloat16).reshape(-1, self.in_features))
        y = int8_gemm(qx, x_scale, self.weight, self.weight_scale, self.bias, self.out_features, self.gelu)
        return tuple(y[i].view(*x.shape[:-1], self.out_features) for i in range(self.parts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)[0]


# ----------------------------------------------------------------------------------------------
# MXFP8 (e4m3 values, power-of-two e8m0 scale per 32 elements along K)
# ----------------------------------------------------------------------------------------------

@triton.jit
def _mxfp8_quantize_kernel(x_ptr, q_ptr, s_ptr, cols, stride_m, col_tiles, BLOCK: tl.constexpr):
    # One program per row. Each block of 32 gets the smallest power-of-two scale 2^e with
    # max|block| / 2^e <= 448, so no value saturates; the e8m0 byte is e + 127. Scales are stored in
    # the cuBLAS block-scale layout: 128 x 4 tiles in row-major tile order, each as [32, 4, 4] =
    # (row % 32, row // 32 % 4, col % 4).
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * stride_m + offs, mask=offs < cols, other=0.0).to(tl.float32)
    xb = tl.reshape(x, (BLOCK // 32, 32))
    amax = tl.max(tl.abs(xb), axis=1)
    e = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0))
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    q = (xb * tl.exp2(-e)[:, None]).to(tl.float8e4nv)
    tl.store(q_ptr + row * cols + offs, tl.reshape(q, (BLOCK,)), mask=offs < cols)
    c = tl.arange(0, BLOCK // 32)
    off = ((row // 128) * col_tiles + c // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + c % 4
    tl.store(s_ptr + off, (e + 127.0).to(tl.uint8), mask=c < cols // 32)


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [M, K] (unit column stride, K % 128 == 0) -> (e4m3 [M, K], swizzled e8m0 scales)."""
    rows, cols = x.shape
    if x.dtype != torch.bfloat16 or x.stride(1) != 1 or cols % 128 or cols > _MAX_FEATURES:
        raise ValueError("MXFP8 quantizer expects BF16 [M, K] with K % 128 == 0 and unit column stride")
    col_tiles = cols // 128
    q = torch.empty((rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    # Zeroed: the layout pads the rows to 128, and padded scales must not hold garbage (0xFF is NaN).
    scales = torch.zeros(triton.cdiv(rows, 128) * 128 * col_tiles * 4, device=x.device, dtype=torch.uint8)
    block = triton.next_power_of_2(cols)
    with torch.cuda.device(x.device):
        _mxfp8_quantize_kernel[(rows,)](x, q, scales, cols, x.stride(0), col_tiles, BLOCK=block,
                                        num_warps=8 if block >= 8192 else 4)
    return q, scales.view(torch.float8_e8m0fnu)


class MXFP8Linear(nn.Module):
    """One or more frozen linears that read the same input, as one cuBLAS MXFP8 GEMM (W8A8).

    `project` returns one contiguous output per linear.
    """

    def __init__(self, linears: list[nn.Linear]):
        super().__init__()
        self.in_features = linears[0].in_features
        self.out_features = linears[0].out_features
        if any(linear.in_features != self.in_features or linear.out_features != self.out_features
               or linear.bias is None for linear in linears):
            raise ValueError("fused linears need the same shape and a bias")
        self.parts = len(linears)
        weight, scale = quantize_mxfp8(torch.cat([linear.weight.detach() for linear in linears]).to(torch.bfloat16).contiguous())
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", torch.cat([linear.bias.detach() for linear in linears]).to(torch.bfloat16))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, parts={self.parts}"

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from torch.nn.functional import ScalingType, SwizzleType, scaled_mm
        q, scale = quantize_mxfp8(x.to(torch.bfloat16).reshape(-1, self.in_features))
        y = scaled_mm(q, self.weight.t(), scale, ScalingType.BlockWise1x32, self.weight_scale, ScalingType.BlockWise1x32,
                      swizzle_a=SwizzleType.SWIZZLE_32_4_4, swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                      bias=self.bias, output_dtype=torch.bfloat16)
        y = y.view(*x.shape[:-1], self.parts * self.out_features)
        if self.parts == 1:
            return (y,)
        # The fused block expects contiguous q / k / v.
        return tuple(y[..., i * self.out_features:(i + 1) * self.out_features].contiguous() for i in range(self.parts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)[0]


class _Fused(nn.Module):
    """FP8 or BF16 linears behind the common `project` interface."""

    def __init__(self, linears: list[nn.Module], fp8: bool):
        super().__init__()
        self.linears = nn.ModuleList(FP8Linear(linear).eval() if fp8 else linear for linear in linears)
        self.fp8 = fp8

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.fp8:
            return project_shared_input(x, tuple(self.linears))
        return tuple(linear(x) for linear in self.linears)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)[0]


def build_linear(fmt: str, linears: list[nn.Linear]) -> nn.Module:
    if fmt == "int8":
        return Int8Linear(linears).eval()
    if fmt == "mxfp8":
        return MXFP8Linear(linears).eval()
    if fmt in ("fp8", "bf16"):
        return _Fused(linears, fp8=fmt == "fp8").eval()
    raise ValueError(f"unknown linear format {fmt!r}")


def convert_block_linears(block: nn.Module, formats: dict) -> None:
    """Replace the linears of one DiT block (on the GPU) in place, by role (see config).

    Self-attention q / k / v become one fused ``qkv`` module and are removed. The cross-attention
    k / v only project the prompt and stay BF16, as do embeddings and the head.
    """
    attn, cross = block.self_attn, block.cross_attn
    attn.qkv = build_linear(formats["self_qkv"], [attn.q, attn.k, attn.v])
    del attn.q, attn.k, attn.v
    attn.o = build_linear(formats["self_o"], [attn.o])
    cross.q = build_linear(formats["cross_q"], [cross.q])
    cross.o = build_linear(formats["cross_o"], [cross.o])
    if formats["ffn_in"] == "int8":
        # The tanh-GELU between the feed-forward linears runs in the first one's GEMM epilogue.
        block.ffn[0], block.ffn[1] = Int8Linear([block.ffn[0]], gelu=True).eval(), nn.Identity()
    else:
        block.ffn[0] = build_linear(formats["ffn_in"], [block.ffn[0]])
    block.ffn[2] = build_linear(formats["ffn_out"], [block.ffn[2]])
