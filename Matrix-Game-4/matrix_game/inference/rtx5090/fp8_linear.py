"""FP8 (e4m3) weight + activation linears for the DiT blocks.

Scheme (post-training, inference only), following the tensor-granularity FP8 inference path of
FastVideo (https://github.com/hao-ai-lab/FastVideo, Apache-2.0), built on ``torch._scaled_mm``:

* What is quantized: the ten linears of a transformer block -- ``self_attn.{q,k,v,o}``,
  ``cross_attn.{q,k,v,o}``, ``ffn.0`` and ``ffn.2`` -- in every block. Embeddings and the head stay
  BF16.
* Weights: one scale per matrix, ``scale = max|W| / 448`` (448 is the largest finite e4m3 value), and
  ``W_q = round_e4m3(W / scale)``. Quantized once at build time.
* Activations: one scale per call, ``scale = max|x| / 448`` over the whole input, computed on the
  device by two small Triton kernels: per-row maxima, then one that reduces them to the scale,
  divides, saturates to +-448 and rounds to e4m3. Nothing is synchronised with the host, so the
  whole layer can be captured in a CUDA graph.
* Product: one FP8 GEMM that accumulates in FP32, applies both scales and adds the BF16 bias in its
  epilogue, then rounds to BF16 once.

The q, k and v projections of an attention layer read the same activation, so
``project_shared_input`` quantizes it once and runs three GEMMs on the shared FP8 tensor. The
activation scale depends only on the input, which makes this bit-identical to three separate calls.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn

_E4M3_MAX = 448.0
_MIN_SCALE = 1.0 / (_E4M3_MAX * 512.0)  # floor for both scales: an all-zero tensor must not divide by zero
_MAX_FEATURES = 32768                   # one Triton program reduces / quantizes a whole row


# ----------------------------------------------------------------------------------------------
# Activation quantization kernels (one program per row of the [M, K] activation)
# ----------------------------------------------------------------------------------------------

@triton.jit
def _row_absmax(x_ptr, out_ptr, cols, stride_m, stride_k, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * stride_m + offs * stride_k, mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.max(tl.abs(x), axis=0))


@triton.jit
def _scale_quantize_rows(x_ptr, y_ptr, max_ptr, scale_ptr, rows, cols, stride_m, stride_k,
                         BLOCK: tl.constexpr, ROWS: tl.constexpr, MIN_SCALE: tl.constexpr):
    # Scale and quantisation in one launch: every row program reduces the row maxima to the tensor
    # scale itself (the same value in all programs), then divides, saturates and rounds its row.
    row = tl.program_id(0)
    index = tl.arange(0, ROWS)
    values = tl.load(max_ptr + index, index < rows, 0.0)
    scale = tl.maximum(tl.max(values, 0) / 448.0, MIN_SCALE)
    if row == 0:
        tl.store(scale_ptr, scale)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * stride_m + offs * stride_k, mask=mask, other=0.0)
    q = tl.minimum(tl.maximum(x.to(tl.float32) / scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(y_ptr + row * cols + offs, q, mask=mask)


def quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 ``[M, K]`` activation; returns (e4m3 ``[M, K]``, FP32 scale ``[1]``).

    ``x`` may be strided (the kernels index through its strides); the result is contiguous.
    The scale stays on the device: ``x ~= fp8 * scale``.
    """
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("activation quantizer expects a CUDA BF16 [M, K] tensor")
    rows, cols = x.shape
    if rows < 1 or not 0 < cols <= _MAX_FEATURES:
        raise ValueError(f"activation quantizer needs M > 0 and 0 < K <= {_MAX_FEATURES}")
    block = max(16, triton.next_power_of_2(cols))
    row_max = torch.empty((rows,), device=x.device, dtype=torch.float32)
    scale = torch.empty((1,), device=x.device, dtype=torch.float32)
    y = torch.empty((rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    with torch.cuda.device(x.device):
        _row_absmax[(rows,)](x, row_max, cols, x.stride(0), x.stride(1), BLOCK=block)
        _scale_quantize_rows[(rows,)](x, y, row_max, scale, rows, cols, x.stride(0), x.stride(1), BLOCK=block,
                                      ROWS=triton.next_power_of_2(rows), MIN_SCALE=_MIN_SCALE)
    return y, scale


# ----------------------------------------------------------------------------------------------
# Layer
# ----------------------------------------------------------------------------------------------

def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[N, K]`` weight -> (e4m3 ``[N, K]``, FP32 scale ``[1]``), computed on the CPU in FP32."""
    weight = weight.detach().cpu().float()
    scale = torch.clamp(weight.abs().max() / _E4M3_MAX, min=_MIN_SCALE)
    return (weight / scale).to(torch.float8_e4m3fn), scale.reshape(1)


class FP8Linear(nn.Module):
    """Frozen e4m3 weight with a per-tensor scale; the activation is quantized on every call.

    ``weight`` is stored as the ``[K, N]`` transposed view of the row-major ``[N, K]`` matrix, i.e.
    the column-major second operand ``torch._scaled_mm`` expects. Move this module between devices
    only: a dtype cast (``.to(dtype)``, ``.half()``, ``.float()``) would destroy the FP8 storage.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.out_features, self.in_features = linear.weight.shape
        if self.in_features % 16 or self.out_features % 16:
            raise ValueError("FP8Linear requires in/out features divisible by 16")
        device = linear.weight.device
        weight, scale = _quantize_weight(linear.weight)
        self.weight = nn.Parameter(weight.to(device).t(), requires_grad=False)
        self.register_buffer("weight_scale", scale.to(device))
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().to(torch.bfloat16), requires_grad=False)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def _check(self, x: torch.Tensor) -> None:
        if self.weight.dtype != torch.float8_e4m3fn or self.weight_scale.dtype != torch.float32:
            raise RuntimeError("FP8Linear storage was cast; move this module by device only")
        if x.device != self.weight.device or not x.is_cuda:
            raise RuntimeError("FP8Linear input and weight must share a CUDA device")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dimension {self.in_features}, got {x.shape[-1]}")

    def _gemm(self, qx: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
        # FP32 accumulation, both scales and the bias applied in the GEMM epilogue, one rounding to BF16.
        return torch._scaled_mm(qx, self.weight, x_scale, self.weight_scale,
                                bias=self.bias, out_dtype=torch.bfloat16, use_fast_accum=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        qx, x_scale = quantize_activation(x.to(torch.bfloat16).reshape(-1, self.in_features))
        return self._gemm(qx, x_scale).reshape(*x.shape[:-1], self.out_features)


def project_shared_input(x: torch.Tensor, linears: tuple) -> tuple[torch.Tensor, ...]:
    """Apply several linears to the same input, quantizing the activation only once.

    Used for the q/k/v projections (all ``FP8Linear``): the input is quantized once and each layer
    runs its own GEMM on the shared FP8 tensor. Results equal ``tuple(l(x) for l in linears)``.
    """
    for linear in linears:
        linear._check(x)
    qx, x_scale = quantize_activation(x.to(torch.bfloat16).reshape(-1, x.shape[-1]))
    return tuple(linear._gemm(qx, x_scale).reshape(*x.shape[:-1], linear.out_features) for linear in linears)


# ----------------------------------------------------------------------------------------------
# Model conversion
# ----------------------------------------------------------------------------------------------

_ATTENTION_LINEARS = ("q", "k", "v", "o")
_FFN_LINEARS = (0, 2)


def quantize_block_linears(model: nn.Module) -> int:
    """Replace the block linears of ``model`` (a ``CausalWanModel``) by ``FP8Linear``, in place.

    Weights are quantized from their current (BF16) values and stay on their device. Returns the
    number of replaced linears.
    """
    count = 0
    for block in model.blocks:
        for attention in (block.self_attn, block.cross_attn):
            for name in _ATTENTION_LINEARS:
                setattr(attention, name, FP8Linear(getattr(attention, name)).eval())
                count += 1
        for position in _FFN_LINEARS:
            block.ffn[position] = FP8Linear(block.ffn[position]).eval()
            count += 1
    return count
