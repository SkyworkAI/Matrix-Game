import torch
import torch.nn as nn
from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange
from .attention import FLASH_ATTN_3_AVAILABLE, FLASH_ATTN_2_AVAILABLE
if FLASH_ATTN_3_AVAILABLE:
    import flash_attn_interface as flash_attn_ops
elif FLASH_ATTN_2_AVAILABLE:
    import flash_attn as flash_attn_ops
else:
    flash_attn_ops = None
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from torch.nn.attention.flex_attention import flex_attention
DISABLE_COMPILE = False

class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).to(x.dtype) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

class ActionModule(nn.Module):
    """
    action module from https://arxiv.org/pdf/2501.08325
    """

    def __init__(
        self, 
        mouse_dim_in: int = 2,
        keyboard_dim_in: int = 6,
        hidden_size: int = 128,
        img_hidden_size: int = 1536,
        keyboard_hidden_dim: int = 1024,
        mouse_hidden_dim: int = 1024,
        vae_time_compression_ratio: int = 4, 
        windows_size: int = 3,
        heads_num: int = 16,
        patch_size: list = [1, 2, 2],
        qk_norm: bool = True,
        qkv_bias: bool = False,
        rope_dim_list: list = [8, 28, 28],
        rope_theta = 256,
        mouse_qk_dim_list = [8, 28, 28],
        enable_mouse = True,
        enable_keyboard = True,
        blocks = [],
        local_attn_size = 6,
    ):
        super().__init__()
        self.local_attn_size = local_attn_size
        self.enable_mouse = enable_mouse
        self.enable_keyboard = enable_keyboard
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        if self.enable_keyboard:
            self.keyboard_embed = nn.Sequential(nn.Linear(keyboard_dim_in, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.mouse_qk_dim_list = mouse_qk_dim_list
        self.heads_num = heads_num
        if self.enable_mouse:
            c = mouse_hidden_dim
            self.mouse_mlp = torch.nn.Sequential(
                torch.nn.Linear(mouse_dim_in * vae_time_compression_ratio * windows_size + img_hidden_size, c, bias=True),
                torch.nn.GELU(approximate="tanh"),
                torch.nn.Linear(c, c),
                torch.nn.LayerNorm(c),
            )
            
            head_dim = c // heads_num
            self.t_qkv = nn.Linear(c, c*3, bias=qkv_bias)
            self.img_attn_q_norm = (
                WanRMSNorm(head_dim, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.img_attn_k_norm = (
                WanRMSNorm(head_dim, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.proj_mouse = nn.Linear(c, img_hidden_size, bias=qkv_bias)

        if self.enable_keyboard:
            head_dim_key = keyboard_hidden_dim // heads_num
            self.key_attn_q_norm = (
                WanRMSNorm(head_dim_key, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.key_attn_k_norm = (
                WanRMSNorm(head_dim_key, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            
            self.mouse_attn_q = nn.Linear(img_hidden_size, keyboard_hidden_dim, bias=qkv_bias)
            self.keyboard_attn_kv = nn.Linear(hidden_size * windows_size * vae_time_compression_ratio, keyboard_hidden_dim * 2, bias=qkv_bias)
            self.proj_keyboard = nn.Linear(keyboard_hidden_dim, img_hidden_size, bias=qkv_bias)

        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.patch_size = patch_size
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
        c =  x.shape[2] // patch_size
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def get_rotary_pos_embed(self, video_length, height, width, head_dim, rope_dim_list = None):
        target_ndim = 3
        ndim = 5 - 2
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
        return freqs_cos[-video_length*rope_sizes[1]*rope_sizes[2]//self.patch_size[0]:], freqs_sin[-video_length*rope_sizes[1]*rope_sizes[2]//self.patch_size[0]:]

    def forward(self, x, tt, th, tw, mouse_condition=None, keyboard_condition=None, mouse_cond_memory=None, keyboard_cond_memory=None):
        '''
        hidden_states: B, tt*th*tw, C
        mouse_condition: B, N_frames, C1
        keyboard_condition: B, N_frames, C2
        '''
        B, N_frames, C = keyboard_condition.shape
        assert tt*th*tw == x.shape[1]
        assert (((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0) or (N_frames % self.vae_time_compression_ratio == 0)
        if ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0:
            N_feats = int((N_frames - 1) / self.vae_time_compression_ratio) + 1
        else:
            N_feats = N_frames // self.vae_time_compression_ratio
        num_frame_per_block = tt
        
        if self.enable_mouse and mouse_condition is not None:
            hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=tt, S=th*tw)
            B, N_frames, C = mouse_condition.shape
        else:
            hidden_states = x
        
        pad_t = self.vae_time_compression_ratio * self.windows_size
        if self.enable_mouse and mouse_condition is not None:
            if ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0:
                mouse_condition = torch.cat([mouse_condition[:, 0:1, :].repeat(1, pad_t, 1), mouse_condition], dim=1)
            else:
                mouse_condition = torch.cat([mouse_condition[:, 0:1, :].repeat(1, pad_t-4, 1), mouse_condition], dim=1)
            group_mouse = [mouse_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(N_feats)]
            group_mouse = torch.stack(group_mouse, dim = 1)
            memory_length = 0
            if mouse_cond_memory is not None:
                memory_length = mouse_cond_memory.shape[1]
                mouse_cond_memory = mouse_cond_memory.unsqueeze(2).repeat(1,1,pad_t,1)
                group_mouse = torch.cat([mouse_cond_memory, group_mouse], dim=1)
            group_mouse = group_mouse.unsqueeze(-1).repeat(1, 1, 1, 1, th * tw)
            group_mouse = rearrange(group_mouse, 'b t window d s -> (b s) t (window d)')
            group_mouse = torch.cat([hidden_states, group_mouse], dim = -1)
            group_mouse = self.mouse_mlp(group_mouse)
            mouse_qkv = self.t_qkv(group_mouse)
            q, k, v = rearrange(mouse_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
            q = self.img_attn_q_norm(q).to(v)
            k = self.img_attn_k_norm(k).to(v)        

            if memory_length > 0:
                freqs_cos_memory, freqs_sin_memory = self.get_rotary_pos_embed(memory_length, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list)
                freqs_cis_memory = (freqs_cos_memory, freqs_sin_memory)
                if freqs_cis_memory is not None:
                    qq_memory, kk_memory = apply_rotary_emb(q[:,:memory_length], k[:,:memory_length], freqs_cis_memory, head_first=False)
                    q[:,:memory_length,:], k[:,:memory_length,:] = qq_memory, kk_memory
                
                freqs_cos_pred, freqs_sin_pred = self.get_rotary_pos_embed(tt - memory_length, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list)
                freqs_cis_pred = (freqs_cos_pred, freqs_sin_pred)
                if freqs_cis_pred is not None:
                    qq_pred, kk_pred = apply_rotary_emb(q[:,memory_length:], k[:,memory_length:], freqs_cis_pred, head_first=False)
                    q[:,memory_length:,:], k[:,memory_length:,:] = qq_pred, kk_pred
            else:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed(tt, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list,)
                freqs_cis = (freqs_cos, freqs_sin)
                if freqs_cis is not None:
                    qq, kk = apply_rotary_emb(q, k, freqs_cis, head_first=False)
                    assert (
                        qq.shape == q.shape and kk.shape == k.shape
                    ), f"qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}"
                    q, k = qq, kk

            if flash_attn_ops is not None:
                attn = flash_attn_ops.flash_attn_func(
                        q,
                        k, 
                        v, 
                    )
            else:
                q_pt = q.transpose(1, 2)
                k_pt = k.transpose(1, 2)
                v_pt = v.transpose(1, 2)
                attn = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt).transpose(1, 2).contiguous()
            attn = rearrange(attn, '(b S) T h d -> b (T S) (h d)',b=B)
            hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=B)
            attn = self.proj_mouse(attn)
            hidden_states = hidden_states + attn
        
        if self.enable_keyboard and keyboard_condition is not None:
            if ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0:
                keyboard_condition = torch.cat([keyboard_condition[:, 0:1, :].repeat(1, pad_t, 1), keyboard_condition], dim=1).to(self.keyboard_embed[0].weight.dtype)
            else:
                keyboard_condition = torch.cat([keyboard_condition[:, 0:1, :].repeat(1, pad_t-4, 1), keyboard_condition], dim=1).to(self.keyboard_embed[0].weight.dtype)

            keyboard_condition = self.keyboard_embed(keyboard_condition)
            group_keyboard = [keyboard_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(N_feats)]
            group_keyboard = torch.stack(group_keyboard, dim = 1)
            if keyboard_cond_memory is not None:
                memory_length = keyboard_cond_memory.shape[1]
                keyboard_cond_memory = self.keyboard_embed(keyboard_cond_memory)
                keyboard_cond_memory = keyboard_cond_memory.unsqueeze(2).repeat(1,1,pad_t,1)
                group_keyboard = torch.cat([keyboard_cond_memory, group_keyboard], dim=1)
            group_keyboard = group_keyboard.reshape(shape=(group_keyboard.shape[0],group_keyboard.shape[1],-1))
            mouse_q = self.mouse_attn_q(hidden_states)
            keyboard_kv = self.keyboard_attn_kv(group_keyboard)
            q = rearrange(mouse_q, "B L (H D) -> B L H D",H=self.heads_num)
            k, v = rearrange(keyboard_kv, "B L (K H D) -> K B L H D", K=2, H=self.heads_num)

            q = self.key_attn_q_norm(q).to(v)
            k = self.key_attn_k_norm(k).to(v)
            S=th*tw 
            q = rearrange(q, "B (T S) H D -> (B S) T H D", S=S)
            '''
            memory_length: the length of the memory
            add rope to memory and pred frames separately.
            '''
            if memory_length > 0:
                freqs_cos_memory, freqs_sin_memory = self.get_rotary_pos_embed(memory_length, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list)
                freqs_cis_memory = (freqs_cos_memory, freqs_sin_memory)
                if freqs_cis_memory is not None:
                    qq_memory, kk_memory = apply_rotary_emb(q[:,:memory_length], k[:,:memory_length], freqs_cis_memory, head_first=False)
                    q[:,:memory_length,:], k[:,:memory_length,:] = qq_memory, kk_memory

                freqs_cos_pred, freqs_sin_pred = self.get_rotary_pos_embed(tt - memory_length, self.patch_size[1], self.patch_size[2], k.shape[-1], self.mouse_qk_dim_list)
                freqs_cis_pred = (freqs_cos_pred, freqs_sin_pred)
                if freqs_cis_pred is not None:
                    qq_pred, kk_pred = apply_rotary_emb(q[:,memory_length:], k[:,memory_length:], freqs_cis_pred, head_first=False)
                    q[:,memory_length:,:], k[:,memory_length:,:] = qq_pred, kk_pred
            else:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed(tt, self.patch_size[1], self.patch_size[2], k.shape[-1], self.rope_dim_list)
                freqs_cis = (freqs_cos, freqs_sin)
                if freqs_cis is not None:
                    qq, kk = apply_rotary_emb(q, k, freqs_cis, head_first=False)
                    assert (
                        qq.shape == q.shape and kk.shape == k.shape
                    ), f"img_kk: {qq.shape}, img_q: {q.shape}, img_kk: {kk.shape}, img_k: {k.shape}"
                    q, k = qq, kk
            k = k.repeat(S, 1, 1, 1)
            v = v.repeat(S, 1, 1, 1)
            if flash_attn_ops is not None:
                attn = flash_attn_ops.flash_attn_func(
                        q,
                        k, 
                        v, 
                        causal=False,
                    )
            else:
                q_pt = q.transpose(1, 2)
                k_pt = k.transpose(1, 2)
                v_pt = v.transpose(1, 2)
                attn = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=False).transpose(1, 2).contiguous()
            attn = rearrange(attn, '(B S) T H D -> B (T S) (H D)', S=S)
            attn = self.proj_keyboard(attn)
            hidden_states = hidden_states + attn
        return hidden_states
