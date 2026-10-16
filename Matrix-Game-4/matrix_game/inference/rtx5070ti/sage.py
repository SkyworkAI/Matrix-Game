"""SageAttention self-attention window without the RTX 5090 window's BF16 copy of V (~1.6 GB):
a key range is re-permuted from the V cache into a shared scratch when requantized."""

import torch
import triton

from ..rtx5090_sage.sage_attention import _fused, _key_mean_kernel, _merge_kernel, sm89_compile


class SageWindow:
    """Quantized companion of one layer's self-attention K / V buffers [1, capacity, H, 128]."""

    def __init__(self, parts: int, scratch: dict):
        self.parts = parts
        self.length = 0        # keys per range
        self.valid = False     # the quantized state matches the BF16 buffers
        self.k_int8 = None     # allocated on first use, once the buffer shape is known
        self._scratch = scratch  # {"v": [1, D, H, padded range] BF16}, shared by every window
        self._cache_v = None

    def usable(self, cache_k: torch.Tensor, start: int, end: int, window_start: int) -> bool:
        """True when the window is full: the ranges then tile the whole buffer and the current block
        is the tail of the last range."""
        capacity = cache_k.shape[1]
        return (end == capacity and window_start == 0 and cache_k.shape[0] == 1
                and capacity % self.parts == 0 and (capacity // self.parts) % (end - start) == 0
                and cache_k.shape[3] == 128)

    def _allocate(self, cache_k: torch.Tensor) -> None:
        parts, device = self.parts, cache_k.device
        _, capacity, heads, dim = cache_k.shape
        self.length = length = capacity // parts
        padded = (length + 63) // 64 * 64
        scratch = self._scratch.get("v")
        if scratch is None or scratch.shape != (1, dim, heads, padded) or scratch.device != device:
            self._scratch["v"] = torch.zeros((1, dim, heads, padded), dtype=cache_k.dtype, device=device)
        self.v_fp8 = torch.zeros((parts, dim, heads, padded), dtype=torch.float8_e4m3fn, device=device)
        self.v_scale = torch.ones((parts, heads, dim), dtype=torch.float32, device=device)
        self.k_int8 = torch.zeros((parts, length, heads, dim), dtype=torch.int8, device=device)
        self.k_scale = torch.ones((parts, heads, (length + 63) // 64), dtype=torch.float32, device=device)
        self.key_mean = torch.zeros((parts, heads, dim), dtype=cache_k.dtype, device=device)

    def _quantize_part(self, cache_k: torch.Tensor, part: int) -> None:
        """Requantize one range's K and V from the current BF16 caches."""
        length = self.length
        heads, dim = cache_k.shape[2:]
        keys = cache_k[:, part * length:(part + 1) * length]
        mean = self.key_mean[part:part + 1]
        _key_mean_kernel[(heads, dim // 32)](keys, mean, length, heads, dim, COLS=32, num_warps=4)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(
            keys, mean, self.k_int8[part:part + 1], self.k_scale[part:part + 1], 64, 0)
        scratch = self._scratch["v"]
        _fused.transpose_pad_permute_cuda(self._cache_v[:, part * length:(part + 1) * length], scratch, 0)
        _fused.scale_fuse_quant_cuda(
            scratch, self.v_fp8[part:part + 1], self.v_scale[part:part + 1], length, 2.25, 0)

    def prepare(self, cache_k, cache_v, start, end, window_start, moved_to) -> None:
        """Bring the quantized state up to date before a forward (eager, outside the graph).
        `moved_to` is the position a ring eviction just copied the previous block to, or None."""
        if not self.usable(cache_k, start, end, window_start):
            self.valid = False
            return
        if self.k_int8 is None:
            self._allocate(cache_k)
        self._cache_v = cache_v
        if not self.valid:
            for part in range(self.parts - 1):  # the last range is requantized by the forward itself
                self._quantize_part(cache_k, part)
            self.valid = True
        elif moved_to is not None and moved_to // self.length != self.parts - 1:
            self._quantize_part(cache_k, moved_to // self.length)

    def attention(self, q: torch.Tensor, cache_k: torch.Tensor, sink_rotated: bool = False) -> torch.Tensor:
        """Self-attention of q [1, M, H, D] (BF16) over the whole window; the current block's K / V
        are already in the caches (last range)."""
        parts = self.parts
        _, m, heads, dim = q.shape
        self._quantize_part(cache_k, parts - 1)
        if sink_rotated:  # the sink's keys (first range) were rotated to a new position
            self._quantize_part(cache_k, 0)
        q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
        q_scale = torch.empty((1, heads, (m + 127) // 128 * 4), dtype=torch.float32, device=q.device)
        _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, 128, 32, 0)
        out = torch.empty((parts, m, heads, dim), dtype=q.dtype, device=q.device)
        sm_scale = dim ** -0.5
        lse = sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int8.expand(parts, -1, -1, -1), self.k_int8, self.v_fp8, out,
            q_scale.expand(parts, -1, -1).contiguous(), self.k_scale, self.v_scale,
            0, 0, 2, sm_scale, 1)
        merged = torch.empty((1, m, heads, dim), dtype=q.dtype, device=q.device)
        _merge_kernel[(m, heads)](out, lse, q, self.key_mean, merged, m, heads, dim, sm_scale,
                                  P=parts, BLOCK=triton.next_power_of_2(dim), num_warps=1)
        return merged
