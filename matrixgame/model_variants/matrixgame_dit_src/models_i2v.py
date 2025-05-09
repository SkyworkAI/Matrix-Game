from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from diffusers.loaders import PeftAdapterMixin
from diffusers.models import ModelMixin
from diffusers.utils import is_torch_version
from diffusers.configuration_utils import ConfigMixin, register_to_config
import torch.utils
import torch.utils.checkpoint

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .attenion import attention, parallel_attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate, ckpt_wrapper
from .token_refiner import SingleTokenRefiner
from .motion_module import ActionModule
from .layernorm import FusedLayerNorm

class MMDoubleStreamBlock(nn.Module):
    """
    Refer to (SD3): https://arxiv.org/abs/2403.03206
            (Flux.1): https://github.com/black-forest-labs/flux
            (HunyuanVideo): https://github.com/Tencent/HunyuanVideo
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        action_config: dict = {},
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = FusedLayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, memory_efficient=True
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )
        self.img_norm2 = FusedLayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, memory_efficient=True
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        # self.txt_norm1 = nn.LayerNorm(
        #     hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        # )
        self.txt_norm1 = FusedLayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, memory_efficient=True
        )
        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        # self.txt_norm2 = nn.LayerNorm(
        #     hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        # )
        self.txt_norm2 = FusedLayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, memory_efficient=True
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        if len(action_config) != 0:
            self.action_model = ActionModule(**action_config)
        else:
            self.action_model = None
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward_v1(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        kwargs = None,
        mouse_condition = None,
        keyboard_condition = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if condition_type == "token_replace":
            img_mod1, token_replace_img_mod1 = self.img_mod(vec, condition_type=condition_type, \
                                                            token_replace_vec=token_replace_vec)
            (img_mod1_shift,
             img_mod1_scale,
             img_mod1_gate,
             img_mod2_shift,
             img_mod2_scale,
             img_mod2_gate) = img_mod1.chunk(6, dim=-1)
            (tr_img_mod1_shift,
             tr_img_mod1_scale,
             tr_img_mod1_gate,
             tr_img_mod2_shift,
             tr_img_mod2_scale,
             tr_img_mod2_gate) = token_replace_img_mod1.chunk(6, dim=-1)
        else:
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.img_mod(vec).chunk(6, dim=-1)

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        if condition_type == "token_replace":
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale, condition_type=condition_type,
                tr_shift=tr_img_mod1_shift, tr_scale=tr_img_mod1_scale,
                frist_frame_token_num=frist_frame_token_num
            )
        else:
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
            )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        
        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=img_k.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )
            
        # attention computation end

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
        if (mouse_condition is not None) and (keyboard_condition is not None):
            mouse_condition.to(img_attn.device, img_attn.dtype)
            keyboard_condition.to(img_attn.device, img_attn.dtype)
            img_attn = self.action_model(mouse_condition,keyboard_condition,img_attn,**kwargs)

        # Calculate the img bloks.
        if condition_type == "token_replace":
            img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate, condition_type=condition_type,
                                   tr_gate=tr_img_mod1_gate, frist_frame_token_num=frist_frame_token_num)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale, condition_type=condition_type,
                        tr_shift=tr_img_mod2_shift, tr_scale=tr_img_mod2_scale, frist_frame_token_num=frist_frame_token_num
                    )
                ),
                gate=img_mod2_gate, condition_type=condition_type,
                tr_gate=tr_img_mod2_gate, frist_frame_token_num=frist_frame_token_num
            )
        else:
            img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                    )
                ),
                gate=img_mod2_gate,
            )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt
    
    def pre_attention(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
        ):    
        if condition_type == "token_replace":
            img_mod1, token_replace_img_mod1 = self.img_mod(vec, condition_type=condition_type, \
                                                            token_replace_vec=token_replace_vec)
            (img_mod1_shift,
             img_mod1_scale,
             img_mod1_gate,
             img_mod2_shift,
             img_mod2_scale,
             img_mod2_gate) = img_mod1.chunk(6, dim=-1)
            (tr_img_mod1_shift,
             tr_img_mod1_scale,
             tr_img_mod1_gate,
             tr_img_mod2_shift,
             tr_img_mod2_scale,
             tr_img_mod2_gate) = token_replace_img_mod1.chunk(6, dim=-1)
        else:
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.img_mod(vec).chunk(6, dim=-1)

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        if condition_type == "token_replace":
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale, condition_type=condition_type,
                tr_shift=tr_img_mod1_shift, tr_scale=tr_img_mod1_scale,
                frist_frame_token_num=frist_frame_token_num
            )
        else:
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
            )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        
            
        img_mod2 = (img_mod2_shift, img_mod2_scale, img_mod2_gate)
        txt_mod2 = (txt_mod2_shift, txt_mod2_scale, txt_mod2_gate,)
        if condition_type == "token_replace":
            tr_mod1  = (tr_img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate)
        else:
            tr_mod1 = None
            
        return q, k, v, img_q, img_k, img_mod1_gate, txt_mod1_gate, img_mod2, txt_mod2, tr_mod1
    
    def post_attention(self, img, txt, attn, img_mod1_gate, txt_mod1_gate,  img_mod2, txt_mod2, tr_mod1,
                       kwargs = None, mouse_condition = None, keyboard_condition = None, condition_type: str = None, 
                       frist_frame_token_num: int = None):
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2
        
        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
        if (mouse_condition is not None) and (keyboard_condition is not None):
            mouse_condition.to(img_attn.device, img_attn.dtype)
            keyboard_condition.to(img_attn.device, img_attn.dtype)
            img_attn = self.action_model(mouse_condition,keyboard_condition,img_attn,**kwargs)

        # Calculate the img bloks.
        if condition_type == "token_replace":
            tr_img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate = tr_mod1
            img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate, condition_type=condition_type,
                                   tr_gate=tr_img_mod1_gate, frist_frame_token_num=frist_frame_token_num)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale, condition_type=condition_type,
                        tr_shift=tr_img_mod2_shift, tr_scale=tr_img_mod2_scale, frist_frame_token_num=frist_frame_token_num
                    )
                ),
                gate=img_mod2_gate, condition_type=condition_type,
                tr_gate=tr_img_mod2_gate, frist_frame_token_num=frist_frame_token_num
            )
        else:
            img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                    )
                ),
                gate=img_mod2_gate,
            )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )
        # print(f"image shape:{img.shape}, txt shape:{txt.shape}")
 
        return img, txt
    
    def forward_v2(self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        kwargs = None,
        mouse_condition = None,
        keyboard_condition = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None):
        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
        q, k, v, img_q, img_k, img_mod1_gate, txt_mod1_gate, img_mod2, txt_mod2, tr_mod1 = torch.utils.checkpoint.checkpoint(self.pre_attention, img, txt, vec, freqs_cis, condition_type, token_replace_vec, frist_frame_token_num, **ckpt_kwargs)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=img_k.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )
            
        return  torch.utils.checkpoint.checkpoint(
            self.post_attention, img, txt, attn, img_mod1_gate, txt_mod1_gate, img_mod2, txt_mod2, tr_mod1, kwargs,
            mouse_condition, keyboard_condition, condition_type, frist_frame_token_num, **ckpt_kwargs)

    def forward(self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        kwargs = None,
        mouse_condition = None,
        keyboard_condition = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
        no_attn_recompute: bool = False):
        if no_attn_recompute:
            return self.forward_v2(img, txt, vec, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis, kwargs,
              mouse_condition, keyboard_condition, condition_type, token_replace_vec, frist_frame_token_num)
        else:
            return self.forward_v1(img, txt, vec, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis, kwargs,
              mouse_condition, keyboard_condition, condition_type, token_replace_vec, frist_frame_token_num)


class MMSingleStreamBlock(nn.Module):
    """
    Refer to (SD3): https://arxiv.org/abs/2403.03206
            (Flux.1): https://github.com/black-forest-labs/flux
            (HunyuanVideo): https://github.com/Tencent/HunyuanVideo
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        action_config: dict = {},
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        if len(action_config) != 0:
            self.action_model = ActionModule(**action_config)
        else:
            self.action_model = None

        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward_v1(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        kwargs = None,
        mouse_condition = None,
        keyboard_condition = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> torch.Tensor:
        if condition_type == "token_replace":
            mod, tr_mod = self.modulation(vec,
                                          condition_type=condition_type,
                                          token_replace_vec=token_replace_vec)
            (mod_shift,
             mod_scale,
             mod_gate) = mod.chunk(3, dim=-1)
            (tr_mod_shift,
             tr_mod_scale,
             tr_mod_gate) = tr_mod.chunk(3, dim=-1)

        else:
            mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        if condition_type == "token_replace":
            x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale, condition_type=condition_type,
                             tr_shift=tr_mod_shift, tr_scale=tr_mod_scale, frist_frame_token_num=frist_frame_token_num)
        else:
            x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        
        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )
        # attention computation end
        if (mouse_condition is not None) and (keyboard_condition is not None):
            image_seq_len = reduce(lambda x, y: x * y,  kwargs.values())
            mouse_condition.to(attn.device, attn.dtype)
            keyboard_condition.to(attn.device, attn.dtype)
            # attn[:,:image_seq_len,...] = self.action_model(mouse_condition,keyboard_condition,attn[:,:image_seq_len,...],**kwargs)
            img_attn, txt_attn = attn[:,:image_seq_len,...], attn[:,image_seq_len:,...]
            img_attn = self.action_model(mouse_condition,keyboard_condition, img_attn, **kwargs)
            attn = torch.cat([img_attn, txt_attn],dim = 1)
        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        if condition_type == "token_replace":
            output = x + apply_gate(output, gate=mod_gate, condition_type=condition_type,
                                    tr_gate=tr_mod_gate, frist_frame_token_num=frist_frame_token_num)
            return output
        else:
            return x + apply_gate(output, gate=mod_gate)

    def pre_attention(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> torch.Tensor:
        if condition_type == "token_replace":
            mod, tr_mod = self.modulation(vec,
                                          condition_type=condition_type,
                                          token_replace_vec=token_replace_vec)
            (mod_shift,
             mod_scale,
             mod_gate) = mod.chunk(3, dim=-1)
            (tr_mod_shift,
             tr_mod_scale,
             tr_mod_gate) = tr_mod.chunk(3, dim=-1)

        else:
            mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        if condition_type == "token_replace":
            x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale, condition_type=condition_type,
                             tr_shift=tr_mod_shift, tr_scale=tr_mod_scale, frist_frame_token_num=frist_frame_token_num)
        else:
            x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)
            
        return q, k, v, img_q, img_k, mod_gate, mlp, tr_mod_gate
    
    def post_attention(self, x,  attn, mlp, mod_gate, tr_mod_gate, kwargs = None, mouse_condition = None,
        keyboard_condition = None, condition_type: str = None, token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None):
        # attention computation end
        if (mouse_condition is not None) and (keyboard_condition is not None):
            image_seq_len = reduce(lambda x, y: x * y,  kwargs.values())
            mouse_condition.to(attn.device, attn.dtype)
            keyboard_condition.to(attn.device, attn.dtype)
            # attn[:,:image_seq_len,...] = self.action_model(mouse_condition,keyboard_condition,attn[:,:image_seq_len,...],**kwargs)
            img_attn, txt_attn = attn[:,:image_seq_len,...], attn[:,image_seq_len:,...]
            img_attn = self.action_model(mouse_condition,keyboard_condition, img_attn, **kwargs)
            attn = torch.cat([img_attn, txt_attn],dim = 1)
        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        if condition_type == "token_replace":
            output = x + apply_gate(output, gate=mod_gate, condition_type=condition_type,
                                    tr_gate=tr_mod_gate, frist_frame_token_num=frist_frame_token_num)
            return output
        else:
            return x + apply_gate(output, gate=mod_gate)
        
        
    def forward_v2(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        kwargs = None,
        mouse_condition = None,
        keyboard_condition = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> torch.Tensor:
        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
        q, k, v, img_q, img_k, mod_gate, mlp, tr_mod_gate = torch.utils.checkpoint.checkpoint(
            self.pre_attention, x, vec, txt_len, freqs_cis, condition_type, token_replace_vec, frist_frame_token_num, **ckpt_kwargs)
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        
        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )

        return torch.utils.checkpoint.checkpoint(
            self.post_attention, x, attn, mlp, mod_gate, tr_mod_gate, kwargs, mouse_condition, keyboard_condition, condition_type, token_replace_vec, frist_frame_token_num, **ckpt_kwargs)
    
    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        kwargs = None,
        mouse_condition = None,
        keyboard_condition = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
        no_attn_recompute: bool = False,
    ) -> torch.Tensor:
        if no_attn_recompute:
            return self.forward_v2(x, vec, txt_len, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis, kwargs,
        mouse_condition, keyboard_condition, condition_type, token_replace_vec, frist_frame_token_num)
        else:
            return self.forward_v1(x, vec, txt_len, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis, kwargs,
        mouse_condition, keyboard_condition, condition_type, token_replace_vec, frist_frame_token_num)
              


class MGVideoDiffusionTransformerI2V(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    Refer to (SD3): https://arxiv.org/abs/2403.03206
            (Flux.1): https://github.com/black-forest-labs/flux
            (HunyuanVideo): https://github.com/Tencent/HunyuanVideo
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        text_states_dim,
        text_states_dim_2,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        rope_theta = 256,
        action_config = {},
        i2v_condition_type = 'token_replace',
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.i2v_condition_type = i2v_condition_type

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2

    
        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )


        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    action_config = action_config,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    action_config = action_config,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.gradient_checkpointing = False
        self.gradient_checkpoint_layers = -1
        self.single_stream_block_no_attn_recompute_layers = 0
        self.double_stream_block_no_attn_recompute_layers = 0
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value
        
    def _set_recompute_layer(self, args):
        self.single_stream_block_no_attn_recompute_layers = args.single_stream_block_no_attn_recompute_layers
        self.double_stream_block_no_attn_recompute_layers = args.double_stream_block_no_attn_recompute_layers


    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        # 884 
        latents_size = [video_length, height, width]

        if isinstance(self.config.patch_size, int):
            assert all(s % self.config.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.config.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.config.patch_size for s in latents_size]
        elif isinstance(self.config.patch_size, list):
            assert all(
                s % self.config.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.config.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.config.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.config.hidden_size // self.config.heads_num
        rope_dim_list = self.config.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.config.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,  # Should be in range(0, 1000).
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,  # Now we don't use it.
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        mouse_condition = None,
        keyboard_condition = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        x = hidden_states
        t = timestep
        text_states, text_states_2 = encoder_hidden_states
        text_mask, test_mask_2 = encoder_attention_mask
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(ot, oh, ow)
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)
        if self.i2v_condition_type == "token_replace":
            token_replace_t = torch.zeros_like(t)
            token_replace_vec = self.time_in(token_replace_t)
            frist_frame_token_num = th * tw
        else:
            token_replace_vec = None
            frist_frame_token_num = None

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        self.single_stream_block_no_attn_recompute_layers = 0
        if img_seq_len < 20 * 1024:
            self.single_stream_block_no_attn_recompute_layers = 10
        elif img_seq_len < 30 * 1024:
            self.single_stream_block_no_attn_recompute_layers = 10
        elif img_seq_len < 40 * 1024:
            self.single_stream_block_no_attn_recompute_layers = 5
        # --------------------- Pass through DiT blocks ------------------------
        for i, block in enumerate(self.double_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing and i >= self.double_stream_block_no_attn_recompute_layers:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                    "th":hidden_states.shape[3] // self.patch_size[1],
                    "tw":hidden_states.shape[4] // self.patch_size[2]}
                img, txt = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                    image_kwargs,
                    mouse_condition,
                    keyboard_condition,
                    self.i2v_condition_type,
                    token_replace_vec,
                    frist_frame_token_num,
                    False,
                    **ckpt_kwargs,
                )
            else:
                image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                    "th":hidden_states.shape[3] // self.patch_size[1],
                    "tw":hidden_states.shape[4] // self.patch_size[2]}
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                    image_kwargs,
                    mouse_condition,
                    keyboard_condition,
                    self.i2v_condition_type,
                    token_replace_vec,
                    frist_frame_token_num,
                    True,
                ]

                img, txt = block(*double_block_args)

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for i, block in enumerate(self.single_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing  and i >= self.single_stream_block_no_attn_recompute_layers:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                        "th":hidden_states.shape[3] // self.patch_size[1],
                        "tw":hidden_states.shape[4] // self.patch_size[2]}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                        image_kwargs,
                        mouse_condition,
                        keyboard_condition,
                        self.i2v_condition_type,
                        token_replace_vec,
                        frist_frame_token_num,
                        False,
                        **ckpt_kwargs,
                    )
                else:
                    image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                        "th":hidden_states.shape[3] // self.patch_size[1],
                        "tw":hidden_states.shape[4] // self.patch_size[2]}
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                        image_kwargs,
                        mouse_condition,
                        keyboard_condition,
                        self.i2v_condition_type,
                        token_replace_vec,
                        frist_frame_token_num,
                        True,
                    ]

                    x = block(*single_block_args)

        img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return (img,)
        
    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts

    def set_input_tensor(self, input_tensor):
        pass


