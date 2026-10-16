"""One causal DiT block on the fused path: the function a block-stack CUDA graph captures.

The arithmetic is that of ``CausalAttentionBlock.forward`` in ``model/causal_model.py``. The
GEMMs and the two attention calls are kept as separate launches; the element-wise / normalisation
chains between them are single Triton kernels (see ``kernels.py``), which round to BF16 at the
same points as the unfused chain. Everything is launched on the current stream without any host
synchronisation, and the function reads no host state besides its arguments, so a captured call
can be replayed for any input values of the same shapes and the same KV placement.
"""

import torch

from .attention import attention
from .fp8_linear import project_shared_input
from .kernels import gate_ln_affine, gate_residual, ln_modulate, qk_rms_rope, rms_norm



def fused_block(block, x: torch.Tensor, e: torch.Tensor, rotations: torch.Tensor,
                cache_k: torch.Tensor, cache_v: torch.Tensor, start: int, end: int, window_start: int,
                cross_k: torch.Tensor, cross_v: torch.Tensor, self_attention=None,
                stop_after_kv: bool = False, sink_k: torch.Tensor | None = None,
                sink_rotations: torch.Tensor | None = None, project_qkv=None,
                cross_attention=None) -> torch.Tensor:
    """
    block       CausalAttentionBlock with FP8 linears
    x           BF16 [B, S, dim], contiguous; S = F * frame_tokens
    e           BF16 [B, F, 6, dim] per-frame timestep modulation (before adding block.modulation)
    rotations   [S, head_dim / 2, 2] RoPE (cos, sin) table of this block's absolute positions
    cache_k/v   BF16 [B, capacity, heads, head_dim]; this block's K/V are written to [start, end)
                and attention reads the keys / values in [window_start, end)
    cross_k/v   BF16 [B, text_len, heads, head_dim] text keys / values (read only)
    self_attention  optional replacement for the self-attention call (see below)
    stop_after_kv   return right after this block's K/V are in the cache (its output is not needed)
    sink_k      BF16 [B, frame_tokens, dim] projected (not yet normalised or rotated) keys of the sink
                frame, written when the block at token 0 is the sink
    sink_rotations  RoPE table of the sink's current position for a fixed-distance sink, or None
    project_qkv     optional replacement for the q / k / v projections: (self_attn, z) -> (q, k, v)
    cross_attention optional replacement for the cross-attention call: (q, cross_k, cross_v) -> y
    returns     BF16 [B, S, dim]
    """
    b, s = x.shape[:2]
    modulation = block.modulation

    # Self-attention. z = LN(x) * (1 + e[1]) + e[0]
    attn = block.self_attn
    z, _ = ln_modulate(x, e, modulation, block.norm1.eps, 0, 1)
    # q, k and v read the same activation: it is quantised once for the three FP8 GEMMs.
    if project_qkv is None:
        q, k, v = project_shared_input(z, (attn.q, attn.k, attn.v))
    else:
        q, k, v = project_qkv(attn, z)
    v = v.view(b, s, attn.num_heads, attn.head_dim)
    cache_v[:, start:end].copy_(v)
    # RMSNorm + RoPE of q and k in one launch; the rotated k goes straight into its cache slot.
    q = qk_rms_rope(q, k, attn.norm_q.weight, attn.norm_k.weight, rotations, cache_k, start,
                    attn.num_heads, attn.norm_q.eps)
    if start == 0 and sink_k is not None:
        sink_k.copy_(k)
    if stop_after_kv:
        return x
    if sink_rotations is not None:
        # Fixed-distance sink: its keys are rotated to the position `window` frames before the current
        # block and written over the sink slot; the values stay as cached.
        qk_rms_rope(sink_k, sink_k, attn.norm_k.weight, attn.norm_k.weight, sink_rotations, cache_k, 0,
                    attn.num_heads, attn.norm_k.eps)
    if self_attention is None:
        y = attention(q, cache_k[:, window_start:end], cache_v[:, window_start:end])
    else:
        y = self_attention(q, v, cache_k, cache_v, start, end, window_start, sink_rotations is not None)
    y = attn.o(y.flatten(2))

    # x = x + y * e[2]; z = norm3(x)
    z, x = gate_ln_affine(x, y, e, modulation, 2, block.norm3.weight, block.norm3.bias, block.norm3.eps)

    # Cross-attention over the cached text keys / values.
    attn = block.cross_attn
    q = rms_norm(attn.q(z), attn.norm_q.weight, attn.norm_q.eps)
    q = q.view(b, s, attn.num_heads, attn.head_dim)
    y = attention(q, cross_k, cross_v) if cross_attention is None else cross_attention(q, cross_k, cross_v)
    y = attn.o(y.flatten(2))

    # x = x + y; z = LN(x) * (1 + e[4]) + e[3]
    z, x = ln_modulate(x, e, modulation, block.norm2.eps, 3, 4, residual=y)

    # Feed-forward. x = x + ffn(z) * e[5]
    return gate_residual(x, block.ffn(z), e, modulation, 5)
