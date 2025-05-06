from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange
from flash_attn import flash_attn_func
import torch
import torch.nn as nn
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .attenion import attention, get_cu_seqlens
from .norm_layers import get_norm_layer
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed, get_1d_rotary_pos_embed

from .activation_layers import get_activation_layer
import torch.nn.functional as F


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension `D`
        pos (`torch.Tensor`): 1D tensor of positions with shape `(M,)`

    Returns:
        `torch.Tensor`: Sinusoidal positional embeddings of shape `(M, D)`.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

class Condition_in(nn.Module):
    def __init__(self, in_channels, hidden_size_mid, hidden_size, windows_size, vae_time_compression_ratio , device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.windows_size = windows_size
        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.embd = MLPEmbedder(in_channels, hidden_size_mid, **factory_kwargs)
        self.out = MLP(in_channels = hidden_size_mid * windows_size * vae_time_compression_ratio, hidden_channels = hidden_size * 2 ,out_features = hidden_size, **factory_kwargs)

    def forward(self, x, length):
        x = self.embd(x) # b l d
        # pos embedding
        pos_embed = get_1d_sincos_pos_embed_from_grid(x.shape[-1], torch.arange(x.shape[1]).to(x.device)).unsqueeze(0)
        x = x + pos_embed.repeat(x.shape[0], 1, 1).to(x.dtype)
        # group
        b, num_frames, dim = x.shape
        pad_t = self.vae_time_compression_ratio * (self.windows_size)
        feature_time_dim = feature_time_dim = int((num_frames - 1) / self.vae_time_compression_ratio) + 1
        x = F.pad(x, (0, 0, pad_t, 0), value=0)
        x = [x[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(feature_time_dim)]
        x = torch.stack(x, dim = 1) # b n l d
        x = x.reshape(shape=(x.shape[0],x.shape[1],-1)) # b n l*d
        # out
        x = self.out(x)
        mask = torch.ones((x.shape[0],x.shape[1])).to(device=x.device, dtype=x.dtype)
        padding = length - x.shape[1]
        if padding < 0:
            raise ValueError("action length is too long !")
        mask = F.pad(mask,(0, padding, 0, 0), value=0)
        x =  F.pad(x,(0, 0, 0, padding), value=0)
        return x, mask

class ActionModule(nn.Module):
    """
    action module from https://arxiv.org/pdf/2501.08325
    鼠标控制信号的输入是一个 L*D 的向量
    键盘同样
    """

    def __init__(
        self, 
        mouse_dim_in: int,
        keyboard_dim_in: int,
        hidden_size: int,
        img_hidden_size: int,
        vae_time_compression_ratio:int, 
        windows_size:int,
        heads_num: int = 24,
        patch_size: list = [1, 2, 2],
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        drop: float = 0.0,
        rope_dim_list: list = [16, 56, 56],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        rope_theta = 256,
        mouse_qk_dim_list = [24, 4, 4],
        enable_mouse = True,
        enable_keyboard = False,
    ):
        device = None
        dtype = None
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.enable_mouse = enable_mouse
        self.enable_keyboard = enable_keyboard
        # self.group = 
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        if self.enable_keyboard:
            self.keyboard_embed = MLPEmbedder(in_dim=keyboard_dim_in, hidden_dim=hidden_size, **factory_kwargs)
        mouse_dim_out = img_hidden_size // (patch_size[0]*patch_size[1]*patch_size[2])
        self.mouse_qk_dim_list = mouse_qk_dim_list
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.heads_num = heads_num
        if self.enable_mouse:
            # self.mouse_embed = MLPEmbedder(in_dim=mouse_dim_in, hidden_dim=hidden_size, **factory_kwargs) # b rn c
            self.mouse_mlp = MLP(
                    mouse_dim_in * vae_time_compression_ratio * windows_size + img_hidden_size,
                    out_features=img_hidden_size,
                    act_layer=get_activation_layer(mlp_act_type),
                    bias=True,
                    **factory_kwargs,
                )
            c = img_hidden_size
            head_dim = c // heads_num
            self.t_qkv = nn.Linear(c, c*3, bias=qkv_bias, **factory_kwargs)
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
            self.proj = nn.Linear(c, c, bias=qkv_bias, **factory_kwargs)
            self.drop = nn.Dropout(drop)
        if self.enable_keyboard:
            head_dim_key = img_hidden_size // heads_num
            self.key_attn_q_norm = (
                qk_norm_layer(head_dim_key, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                if qk_norm
                else nn.Identity()
            )
            self.key_attn_k_norm = (
                qk_norm_layer(head_dim_key, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                if qk_norm
                else nn.Identity()
            )
            self.mouse_attn_q = nn.Linear(img_hidden_size, img_hidden_size, bias=qkv_bias, **factory_kwargs)
            self.keyboard_attn_kv = nn.Linear(hidden_size * windows_size * vae_time_compression_ratio, img_hidden_size * 2, bias=qkv_bias, **factory_kwargs)
            self.proj = nn.Linear(img_hidden_size, img_hidden_size, bias=qkv_bias, **factory_kwargs)
            self.drop = nn.Dropout(drop)
        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.patch_size = patch_size
        # self.unpatchify_channels = unpatchify_channels # 16 VAE.config.latent_channels * patch_size ** 2

    def patchify(self, x, patch_size):
        """
        x : (N C T H W)
        """
        pt, ph, pw = self.patch_size
        t, h, w = x.shape[2] //  pt, x.shape[3] // ph, x.shape[4] // pw
        c = x.shape[1]
        x = x.reshape(shape=(x.shape[0], c, t , pt, h , ph, w , pw))
        x = torch.einsum("nctohpwq->nthwcopq", x)
        x = x.reshape(shape=(x.shape[0], t*h*w,  c*pt*ph*pw))
        return x

    def unpatchify(self, x, t, h, w, patch_size):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c =  x.shape[2] // patch_size #self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def get_rotary_pos_embed(self, video_length, height, width, head_dim, rope_dim_list = None):
        target_ndim = 3
        ndim = 5 - 2
        # 884 
        latents_size = [video_length, height, width]

        if isinstance(self.patch_size, int):
            assert all(s % self.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.patch_size for s in latents_size]
        elif isinstance(self.patch_size, list):
            assert all(
                s % self.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def forward(self, mouse_condition, keyboard_condition, x, tt, th, tw):
        '''
        hidden_states: b l 3072
        mouse_condition: b t d
        keyboard_condition:b t d
        '''
        b, num_frames, dim = keyboard_condition.shape
        assert (num_frames + (self.vae_time_compression_ratio - 1)) % self.vae_time_compression_ratio == 0
        feature_time_dim = int((num_frames - 1) / self.vae_time_compression_ratio) + 1
        ## unpathch
        if self.enable_mouse:
            hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=tt, S=th*tw)
            # hidden_states = self.unpatchify(x, tt, th, tw, self.patch_size[0]*self.patch_size[1]*self.patch_size[2])
            # mouse_condition = self.mouse_embed(mouse_condition)  # b t hidden_size
            b, num_frames, dim = mouse_condition.shape
        else:
            hidden_states = x
        # padding
        pad_t = self.vae_time_compression_ratio * (self.windows_size)
        if self.enable_mouse:
            mouse_condition = F.pad(mouse_condition, (0, 0, pad_t, 0))
            group_mouse = [mouse_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(feature_time_dim)]
            group_mouse = torch.stack(group_mouse, dim = 1)
            # group
            group_mouse = torch.stack([group_mouse] * th * tw, dim = -1) # b t win*scale d s
            group_mouse = rearrange(group_mouse, 'b t window d s -> (b s) t (window d)')
            # concat hidden_states, group_mouse
            group_mouse = torch.cat([hidden_states, group_mouse], dim = -1)
            group_mouse = self.mouse_mlp(group_mouse)
            # qkv
            mouse_qkv = self.t_qkv(group_mouse)
            q,k,v = rearrange(mouse_qkv, "B L (K H D) -> K B L H D",K=3,H=self.heads_num) # b l h d
            q = self.img_attn_q_norm(q).to(v)
            k = self.img_attn_k_norm(k).to(v)        
            # rope embd
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(tt, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list)
            freqs_cis = (freqs_cos, freqs_sin)
            if freqs_cis is not None:
                qq, kk = apply_rotary_emb(q, k, freqs_cis, head_first=False)
                assert (
                    qq.shape == q.shape and kk.shape == k.shape
                ), f"qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}"
                q, k = qq, kk
            attn = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=q.shape[-1]**-0.5,
                causal=False,
            )
            # Compute cu_squlens and max_seqlen for flash attention
            # qk norm
            attn = rearrange(attn, '(b S) T h d -> b (T S) (h d)',b=b)
            hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=b)
            attn = self.drop(self.proj(attn))
            hidden_states = hidden_states + attn
        if self.enable_keyboard:
            keyboard_condition = self.keyboard_embed(keyboard_condition) # b t hidden_size
            # add pos embed
            pos_embed = get_1d_sincos_pos_embed_from_grid(keyboard_condition.shape[-1], torch.arange(keyboard_condition.shape[1]).to(keyboard_condition.device)).unsqueeze(0)
            keyboard_condition = keyboard_condition + pos_embed.repeat((keyboard_condition.shape[0], 1, 1)).to(keyboard_condition.dtype)
            keyboard_condition = F.pad(keyboard_condition, (0, 0, pad_t, 0))
            ## group:  b t hidden_size --- > (n + 1) * rw * hidden_size
            group_keyboard = [keyboard_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(feature_time_dim)]
            group_keyboard = torch.stack(group_keyboard, dim = 1)
            group_keyboard = group_keyboard.reshape(shape=(group_keyboard.shape[0],group_keyboard.shape[1],-1))
            ## repeat
        
            # apply corss attn
            mouse_q = self.mouse_attn_q(hidden_states)
            keyboard_kv = self.keyboard_attn_kv(group_keyboard)
            q = rearrange(mouse_q, "B L (H D) -> B L H D",H=self.heads_num)
            k, v = rearrange(keyboard_kv, "B L (K H D) -> K B L H D",K=2,H=self.heads_num)
            # Compute cu_squlens and max_seqlen for flash attention
            # qk norm
            q = self.key_attn_q_norm(q).to(v)
            k = self.key_attn_k_norm(k).to(v)
            # position embed 
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(tt * self.patch_size[0], th * self.patch_size[1], tw * self.patch_size[2], k.shape[-1], self.rope_dim_list)
            freqs_cis = (freqs_cos, freqs_sin)
            if freqs_cis is not None:
                qq, kk = apply_rotary_emb(q, k, freqs_cis, head_first=False)
                assert (
                    qq.shape == q.shape and kk.shape == k.shape
                ), f"img_kk: {qq.shape}, img_q: {q.shape}, img_kk: {kk.shape}, img_k: {k.shape}"
                q, k = qq, kk
            attn = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=q.shape[-1]**-0.5,
                causal=False,
            )
            attn = rearrange(attn, 'B L H D -> B L (H D)')
            attn = self.drop(self.proj(attn))
            hidden_states = hidden_states + attn
        return hidden_states
        
