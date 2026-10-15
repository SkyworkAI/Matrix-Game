"""SageAttention-2 self-attention over a persistent, split ring KV window (RTX 5090, sm_120).

Kernels from SageAttention (https://github.com/thu-ml/SageAttention, Apache-2.0).

``sageattn`` prepares the whole key / value window on every call: key mean, INT8 keys, and the
transpose + pad + permute and FP8 quantisation of V. For one 405-query block over 8505 keys that
preparation costs as much as the attention kernel itself, although 20 of the 21 frames in the
window did not change. With the ring KV window the history stays in place, so:

* The window is ``parts`` equal key ranges launched as one batch (queries broadcast, no copy),
  which fills the device.
* Each range keeps its quantised state next to the BF16 cache: INT8 keys with the range's own key
  mean, V in the kernel's permuted layout, FP8 V with the range's per-channel scale. A range is
  requantised only when one of its frames changes: the last range on every forward (the current
  block lives there) and the range a ring eviction writes to.
* The exact softmax over the window is recovered from each range's log-sum-exp. Subtracting a
  key mean shifts every score of a range by ``q . mean``; that shift is added back per range
  before merging, so ranges may use different means.

A range's quantised state is always exactly what quantising its current BF16 content gives;
nothing is frozen or approximated. Compared with one ``sageattn`` call over the window, key mean
and V scale are per range instead of per window, and the softmax is summed per range and merged.

`SageWindow.prepare` runs eagerly on the host side of a forward; `SageWindow.attention` is
launched without host synchronisation and is what the block-stack CUDA graph captures.
"""

import torch
import triton
import triton.language as tl

try:
    from sageattention import _fused, sm89_compile
except ImportError as error:
    raise ImportError("this runtime requires SageAttention 2.2 built with matrix_game/inference/rtx5090_sage/sageattention.patch") from error

# The self-attention window is split into this many key ranges; 7 divides the 21-frame window.
ATTENTION_PARTS = 7


@triton.jit
def _scatter_v_kernel(src_ptr, dst_ptr, local0, tokens, group0, heads, stride_d, stride_h,
                      D: tl.constexpr):
    # src: [tokens, heads, D] BF16 rows that belong at part-local positions
    # local0 .. local0 + tokens; dst: that part of the permuted V,
    # [D, heads, padded_len].  Sage permutes the sequence inside groups of
    # 16 for its FP8 mma (0 1 4 5 8 9 12 13 2 3 6 7 10 11 14 15), so one
    # program moves one group of 16 tokens of one head as a 16 x D tile.
    group = group0 + tl.program_id(0)
    head = tl.program_id(1)
    r = tl.arange(0, 16)[:, None]
    d = tl.arange(0, D)[None, :]
    t = group * 16 + r
    valid = (t >= local0) & (t < local0 + tokens)
    value = tl.load(src_ptr + ((t - local0) * heads + head) * D + d, mask=valid, other=0.0)
    column = group * 16 + (r // 8) * 2 + ((r // 2) % 4) * 4 + (r % 2)
    tl.store(dst_ptr + d * stride_d + head * stride_h + column, value, mask=valid)

@triton.jit
def _key_mean_kernel(k_ptr, out_ptr, tokens, heads, D, COLS: tl.constexpr):
    # k: [tokens, heads, D] BF16; out: [heads, D] BF16 mean over tokens (FP32 sum).
    head = tl.program_id(0)
    c0 = tl.program_id(1) * COLS
    cols = c0 + tl.arange(0, COLS)[None, :]
    acc = tl.zeros((COLS,), dtype=tl.float32)
    for t0 in range(0, tokens, 64):
        r = t0 + tl.arange(0, 64)[:, None]
        x = tl.load(k_ptr + (r * heads + head) * D + cols, mask=r < tokens, other=0.0)
        acc += tl.sum(x.to(tl.float32), axis=0)
    tl.store(out_ptr + head * D + c0 + tl.arange(0, COLS), (acc / tokens).to(tl.bfloat16))

@triton.jit
def _merge_kernel(out_ptr, lse_ptr, q_ptr, km_ptr, dst_ptr, M, H, D, sm_scale,
                  P: tl.constexpr, BLOCK: tl.constexpr):
    # out: [P, M, H, D] partial outputs; lse: [P, H, M] FP32 log2-sum-exp2 of
    # each part's scores q . (k - km_p); q: [M, H, D]; km: [P, H, D].
    m = tl.program_id(0)
    h = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    q = tl.load(q_ptr + (m * H + h) * D + cols, mask=mask, other=0.0).to(tl.float32)
    top = -float("inf")
    total = 0.0
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for p in tl.static_range(P):
        km = tl.load(km_ptr + (p * H + h) * D + cols, mask=mask, other=0.0).to(tl.float32)
        top = tl.maximum(top, tl.load(lse_ptr + (p * H + h) * M + m) * 0.6931471805599453
                         + tl.sum(q * km, axis=0) * sm_scale)
    for p in tl.static_range(P):
        km = tl.load(km_ptr + (p * H + h) * D + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.exp(tl.load(lse_ptr + (p * H + h) * M + m) * 0.6931471805599453
                   + tl.sum(q * km, axis=0) * sm_scale - top)
        total += w
        acc += w * tl.load(out_ptr + ((p * M + m) * H + h) * D + cols,
                           mask=mask, other=0.0).to(tl.float32)
    tl.store(dst_ptr + (m * H + h) * D + cols, (acc / total).to(dst_ptr.dtype.element_ty),
             mask=mask)


class SageWindow:
    """Quantised companion of one layer's self-attention K/V buffer."""

    def __init__(self, parts: int = ATTENTION_PARTS):
        self.parts = parts
        self.valid = False     # the quantised state matches the BF16 buffers
        self.length = 0        # keys per range
        self.vperm = None      # tensors are allocated on first use, once the buffer shape is known

    def usable(self, cache_k: torch.Tensor, start: int, end: int, window_start: int) -> bool:
        """True when the window is full: the ranges then tile the whole buffer and the current
        block is the tail of the last range."""
        capacity = cache_k.shape[1]
        return (end == capacity and window_start == 0 and cache_k.shape[0] == 1
                and capacity % self.parts == 0 and (capacity // self.parts) % (end - start) == 0
                and cache_k.shape[3] == 128)

    def _allocate(self, cache_k: torch.Tensor) -> None:
        parts, device = self.parts, cache_k.device
        _, capacity, heads, dim = cache_k.shape
        self.length = length = capacity // parts
        padded = (length + 63) // 64 * 64
        self.vperm = torch.zeros((parts, dim, heads, padded), dtype=cache_k.dtype, device=device)
        self.v_fp8 = torch.zeros((parts, dim, heads, padded), dtype=torch.float8_e4m3fn, device=device)
        self.v_scale = torch.ones((parts, heads, dim), dtype=torch.float32, device=device)
        self.k_int8 = torch.zeros((parts, length, heads, dim), dtype=torch.int8, device=device)
        self.k_scale = torch.ones((parts, heads, (length + 63) // 64), dtype=torch.float32, device=device)
        self.key_mean = torch.zeros((parts, heads, dim), dtype=cache_k.dtype, device=device)

    def _scatter(self, source: torch.Tensor, token0: int) -> int:
        """Write V rows `source` [1, tokens, heads, D] at window position `token0` into the permuted V."""
        part, local0 = divmod(token0, self.length)
        tokens, heads, dim = source.shape[1:]
        group0 = local0 // 16
        groups = (local0 + tokens + 15) // 16 - group0
        _scatter_v_kernel[(groups, heads)](
            source, self.vperm[part], local0, tokens, group0, heads, self.vperm.stride(1),
            self.vperm.stride(2), D=dim, num_warps=4)
        return part

    def _quantize_part(self, cache_k: torch.Tensor, part: int) -> None:
        """Requantise one range's K and V from its current BF16 content."""
        length = self.length
        heads, dim = cache_k.shape[2:]
        keys = cache_k[:, part * length:(part + 1) * length]
        mean = self.key_mean[part:part + 1]
        _key_mean_kernel[(heads, dim // 32)](keys, mean, length, heads, dim, COLS=32, num_warps=4)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(
            keys, mean, self.k_int8[part:part + 1], self.k_scale[part:part + 1], 64, 0)
        _fused.scale_fuse_quant_cuda(
            self.vperm[part:part + 1], self.v_fp8[part:part + 1], self.v_scale[part:part + 1], length, 2.25, 0)

    def prepare(self, cache_k: torch.Tensor, cache_v: torch.Tensor, start: int, end: int,
                window_start: int, moved_to: int | None) -> None:
        """Bring the quantised state up to date before a forward (eager, outside the graph).

        `moved_to` is the window position a ring eviction just copied the previous block to
        (None when nothing moved).
        """
        if not self.usable(cache_k, start, end, window_start):
            self.valid = False
            return
        if self.vperm is None:
            self._allocate(cache_k)
        parts, length = self.parts, self.length
        if not self.valid:
            heads, dim = cache_v.shape[2:]
            _fused.transpose_pad_permute_cuda(cache_v.view(parts, length, heads, dim), self.vperm, 0)
            # The last range is requantised by the forward itself.
            for part in range(parts - 1):
                self._quantize_part(cache_k, part)
            self.valid = True
        elif moved_to is not None:
            part = self._scatter(cache_v[:, moved_to:moved_to + end - start], moved_to)
            if part != parts - 1:
                self._quantize_part(cache_k, part)

    def attention(self, q: torch.Tensor, cache_k: torch.Tensor, new_v: torch.Tensor, start: int,
                  sink_rotated: bool = False) -> torch.Tensor:
        """Self-attention of q [1, M, H, D] (BF16) over the whole window.

        `new_v` [1, M, H, D] is the current block's V; its K is already in `cache_k[:, start:]`.
        """
        parts = self.parts
        _, m, heads, dim = q.shape
        self._scatter(new_v, start)
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
