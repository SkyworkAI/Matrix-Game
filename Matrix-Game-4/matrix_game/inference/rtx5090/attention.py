"""Dense attention for the RTX 5090 runtime.

PyTorch's scaled-dot-product attention dispatches BF16 inputs on the RTX 5090 to its built-in
FlashAttention kernels, which already split long key ranges across the device, so self-attention
over the full 21-frame window is a single call. The call launches without host synchronisation and
can be captured in a CUDA graph.
"""

import torch
import torch.nn.functional as F


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Unmasked multi-head attention: q [B, Lq, H, D], k / v [B, Lk, H, D] -> [B, Lq, H, D]."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2).contiguous()
