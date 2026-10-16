"""SageAttention cross-attention; text K / V are quantised once per prompt."""

import torch

from ..rtx5090_sage.sage_attention import _fused, _key_mean_kernel, sm89_compile


class SageText:
    """Quantised companion of one layer's text K / V buffers [1, L, H, D] (L a multiple of 64)."""

    def __init__(self, cache_k: torch.Tensor):
        b, length, heads, dim = cache_k.shape
        if b != 1 or length % 64 or dim != 128:
            raise ValueError("SageText needs [1, L, H, 128] text keys with L a multiple of 64")
        device = cache_k.device
        self.vperm = torch.empty((1, dim, heads, length), dtype=cache_k.dtype, device=device)
        self.v_fp8 = torch.empty((1, dim, heads, length), dtype=torch.float8_e4m3fn, device=device)
        self.v_scale = torch.empty((1, heads, dim), dtype=torch.float32, device=device)
        self.k_int8 = torch.empty((1, length, heads, dim), dtype=torch.int8, device=device)
        self.k_scale = torch.empty((1, heads, length // 64), dtype=torch.float32, device=device)
        self.key_mean = torch.empty((1, heads, dim), dtype=cache_k.dtype, device=device)

    def quantize(self, cache_k: torch.Tensor, cache_v: torch.Tensor) -> None:
        """Requantise from the current BF16 text K / V (eager, after a prompt change)."""
        _, length, heads, dim = cache_k.shape
        _key_mean_kernel[(heads, dim // 32)](cache_k, self.key_mean, length, heads, dim, COLS=32, num_warps=4)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(cache_k, self.key_mean, self.k_int8, self.k_scale, 64, 0)
        _fused.transpose_pad_permute_cuda(cache_v, self.vperm, 0)
        _fused.scale_fuse_quant_cuda(self.vperm, self.v_fp8, self.v_scale, length, 2.25, 0)

    def attention(self, q: torch.Tensor) -> torch.Tensor:
        """q [1, M, H, D] BF16 -> [1, M, H, D]; launched without host synchronisation."""
        _, m, heads, dim = q.shape
        q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
        q_scale = torch.empty((1, heads, (m + 127) // 128 * 4), dtype=torch.float32, device=q.device)
        _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, 128, 32, 0)
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int8, self.k_int8, self.v_fp8, out, q_scale, self.k_scale, self.v_scale, 0, 0, 2, dim ** -0.5, 0)
        return out
