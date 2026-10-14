"""Lossless causal RoPE application backed by a Triton kernel.

The reference implementation rotates pairs after promoting the input to
float64 and multiplying complex tensors.  The Triton kernel deliberately
keeps that promotion and performs the two complex multiply-add expressions in
float64 before storing back to the input dtype.
"""

from __future__ import annotations

import torch


def build_causal_rope_rotations(
    freqs: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    start_frame: int = 0,
) -> torch.Tensor:
    """Expand Wan's three-axis complex RoPE table for a frame-aligned grid.

    ``start_frame`` is the temporal RoPE position of the first frame in this
    grid. Callers using rolling re-RoPE pass the cache slot, rather than the
    source video's absolute frame number.
    """
    if freqs.ndim != 2 or not torch.is_complex(freqs):
        raise ValueError("freqs must be a two-dimensional complex tensor")
    if frames <= 0 or height <= 0 or width <= 0 or start_frame < 0:
        raise ValueError("RoPE grid dimensions must be positive and start non-negative")
    channels = freqs.shape[1]
    height_channels = channels // 3
    width_channels = channels // 3
    temporal_channels = channels - height_channels - width_channels
    temporal, height_freqs, width_freqs = freqs.split(
        [temporal_channels, height_channels, width_channels], dim=1
    )
    if start_frame + frames > temporal.shape[0]:
        raise ValueError("temporal RoPE position exceeds the precomputed frequency table")
    if height > height_freqs.shape[0] or width > width_freqs.shape[0]:
        raise ValueError("spatial RoPE grid exceeds the precomputed frequency table")

    expanded = torch.cat(
        [
            temporal[start_frame:start_frame + frames]
            .view(frames, 1, 1, -1).expand(frames, height, width, -1),
            height_freqs[:height]
            .view(1, height, 1, -1).expand(frames, height, width, -1),
            width_freqs[:width]
            .view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, channels)
    return torch.view_as_real(expanded).contiguous()


def validate_rope_cache_mode(cache: dict, rope_mode: str, local_end: int) -> None:
    """Reject reusing post-RoPE and raw-K cache contents under another mode."""
    if rope_mode not in ("absolute", "rolling"):
        raise ValueError(f"unknown rope_mode: {rope_mode}")
    cached_mode = cache.get("_rope_mode", "absolute")
    if local_end and cached_mode != rope_mode:
        raise ValueError(
            f"cannot switch rope_mode from {cached_mode!r} to {rope_mode!r} "
            "while the KV cache contains entries; clear the KV cache first"
        )
    if rope_mode == "rolling":
        cache["_rope_mode"] = "rolling"
    elif not local_end:
        cache.pop("_rope_mode", None)

try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except ImportError:  # pragma: no cover - exercised only without Triton.
    triton = None
    tl = None
    triton_op = None
    wrap_triton = None


def _apply_causal_rope_cpu(x: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Reference implementation matching the FP64 complex baseline."""
    b, s, n, d = x.shape
    x_complex = torch.view_as_complex(x.double().reshape(b, s, n, d // 2, 2))
    rotation_complex = torch.view_as_complex(rotations)
    rotated = x_complex * rotation_complex.reshape(1, s, 1, d // 2)
    return torch.view_as_real(rotated).flatten(-2).to(dtype=x.dtype)


if triton is not None:

    @triton.jit
    def _causal_rope_kernel(
        x_ptr,
        rotations_ptr,
        out_ptr,
        numel,
        seq_len,
        num_heads,
        head_dim,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel

        # x is contiguous [B, S, N, D].  Each adjacent pair is one complex
        # value and rotations is contiguous [S, D/2, 2] as [cos, sin].
        pair = (offsets % head_dim) // 2
        component = offsets % 2
        token = (offsets // (num_heads * head_dim)) % seq_len
        rotation_offset = token * (head_dim // 2) * 2 + pair * 2

        cos = tl.load(rotations_ptr + rotation_offset, mask=mask, other=0.0)
        sin = tl.load(rotations_ptr + rotation_offset + 1, mask=mask, other=0.0)

        # Keep the baseline's FP64 complex multiplication explicit; the launch
        # disables FMA contraction to preserve the reference rounding.
        real = tl.load(
            x_ptr + offsets - component, mask=mask, other=0
        ).to(tl.float64)
        imag = tl.load(
            x_ptr + offsets - component + 1, mask=mask, other=0
        ).to(tl.float64)
        rotated_real = real * cos - imag * sin
        rotated_imag = real * sin + imag * cos
        result = tl.where(component == 0, rotated_real, rotated_imag)
        # PyTorch's complex FP64 result is converted through the scalar
        # floating implementation before the final half/bfloat16 store.
        tl.store(out_ptr + offsets, result.to(tl.float32), mask=mask)


    @triton_op("videox::causal_rope", mutates_args={})
    def _causal_rope_triton(
        x: torch.Tensor, rotations: torch.Tensor
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        b, s, n, d = x.shape
        numel = out.numel()

        def grid(meta):
            return (triton.cdiv(numel, meta["BLOCK"]),)

        wrap_triton(_causal_rope_kernel)[grid](
            x,
            rotations,
            out,
            numel,
            s,
            n,
            d,
            BLOCK=256,
            enable_fp_fusion=False,
        )
        return out


def apply_causal_rope(x: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Apply causal RoPE to ``x`` using pre-expanded FP64 rotations.

    Args:
        x: Contiguous ``[B, S, N, D]`` tensor with an even ``D``.
        rotations: Contiguous ``[S, D/2, 2]`` float64 tensor containing
            ``[cos, sin]`` for every sequence position and pair.
    """
    if x.ndim != 4 or rotations.ndim != 3:
        raise ValueError("expected x=[B,S,N,D] and rotations=[S,D/2,2]")
    if x.shape[-1] % 2 or rotations.shape[-1] != 2:
        raise ValueError("head dimension and rotation pairs must be even")
    if x.shape[1] != rotations.shape[0] or x.shape[-1] // 2 != rotations.shape[1]:
        raise ValueError("rotation shape does not match x")
    if not x.is_contiguous() or not rotations.is_contiguous():
        raise ValueError("x and rotations must be contiguous")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("x must be float16, bfloat16, or float32")
    if x.device != rotations.device:
        raise ValueError("x and rotations must be on the same device")
    if rotations.dtype != torch.float64:
        raise ValueError("rotations must be float64")

    if x.device.type != "cuda" or triton is None:
        return _apply_causal_rope_cpu(x, rotations)
    return _causal_rope_triton(x, rotations)


__all__ = ["apply_causal_rope"]
