import torch
try:
    import flash_attn_interface
    import os
    # Centralized control for Flash Attention version
    # WAN_FA_VERSION: "2" or "3". Defaults to "3" if available.
    WAN_FA_VERSION = os.getenv("WAN_FA_VERSION", "3")
    
    if WAN_FA_VERSION == "3":
        FLASH_ATTN_3_AVAILABLE = True
    else:
        FLASH_ATTN_3_AVAILABLE = False
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import importlib as _il
    _aiter_mha = _il.import_module('aiter.ops.mha')
    _aiter_flash_attn_varlen = _aiter_mha.flash_attn_varlen_func
    AITER_AVAILABLE = True
except Exception:
    AITER_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except Exception:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    if (version is None or version == 3 or version == "3") and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    elif AITER_AVAILABLE:
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True)
        x = _aiter_flash_attn_varlen(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq, max_seqlen_k=lk,
            softmax_scale=softmax_scale, causal=causal,
        ).unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        raise RuntimeError(
            'No flash attention backend available. Install one of: '
            'flash-attn (NVIDIA), or aiter (AMD ROCm 7.x: pip install aiter)'
        )

    return x.type(out_dtype)


_WARNED_FA_DISABLED = False

def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    global _WARNED_FA_DISABLED
    if version != '0' and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE or AITER_AVAILABLE):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=version,
        )
    else:
        if version == '0':
            if not _WARNED_FA_DISABLED:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print('DEBUG: Flash attention is DISABLED by user, using scaled_dot_product_attention (SDPA).')
                _WARNED_FA_DISABLED = True
        elif q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        
        attn_mask = None
        
        if q.dim() == 3:
            q = q.unsqueeze(0).transpose(1, 2).to(dtype)
            k = k.unsqueeze(0).transpose(1, 2).to(dtype)
            v = v.unsqueeze(0).transpose(1, 2).to(dtype)
            
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)
            
            out = out.transpose(1, 2).squeeze(0).contiguous()
        else:
            q = q.transpose(1, 2).to(dtype)
            k = k.transpose(1, 2).to(dtype)
            v = v.transpose(1, 2).to(dtype)

            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

            out = out.transpose(1, 2).contiguous()
            
        return out
