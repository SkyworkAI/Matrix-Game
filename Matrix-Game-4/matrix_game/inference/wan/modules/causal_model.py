from matrix_game.inference.wan.modules.attention import attention
from matrix_game.inference.sink_policy import check_sink_rope_mode, rotated_sink_position
from matrix_game.inference.wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
try:  # 序列并行（只在教师强制训练里生效，默认关，关时与原实现逐值相同）
    from matrix_game.inference import seq_parallel as _sp
except ImportError:  # 不在仓库根目录下运行的脚本没有 utils 包，按不开序列并行处理
    _sp = None


def _sp_on():
    return _sp is not None and _sp.sp_size() > 1

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    mode="default"
)



def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2
    # 频率表只有 rope_max_len 行（默认 1024），越界时切片会悄悄变短、后面 view 报形状错，这里先给明白的报错
    _check_rope_len(freqs, int(start_frame) + int(grid_sizes[:, 0].max()))

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def _check_rope_len(freqs, need):
    """要用到第 need-1 行（时间维位置最大 need-1）时，频率表行数不够就报错，提示调 rope_max_len。"""
    if need > freqs.shape[0]:
        raise ValueError(f"RoPE 频率表只有 {freqs.shape[0]} 行，要用到第 {need - 1} 个位置；长推理请调大 rope_max_len"
                         f"（CausalWanModel.set_rope_max_len，推理脚本 --rope_max_len，训练配置 rope_max_len）")


def _rope_time_rows(freqs_t, pos):
    """按逐帧位置取时间维 RoPE 行。位置可以为负（记忆潜变量放在 sink 之前，p 小的时候落到负位置）：
    负位置取 conj(freqs[-pos])，即旋转 -pos*theta（RoPE 是复数旋转，e^{-i a} = conj(e^{i a})）。
    全部非负时就是 freqs_t[pos]，与原来逐位相同。"""
    _check_rope_len(freqs_t, int(pos.abs().max()) + 1)   # 位置在 CPU 上时不触发 GPU 同步（新路径都在 CPU 上给位置）
    neg = pos < 0
    pos = pos.to(freqs_t.device)
    if bool(neg.any()):
        rows = freqs_t[pos.abs()]
        return torch.where(neg.to(freqs_t.device).view(-1, 1), torch.conj_physical(rows), rows)
    return freqs_t[pos]


def rope_apply_frames(x, grid_sizes, freqs, frame_pos):
    """与 causal_rope_apply 相同的复数乘法，只是时间维的 RoPE 位置逐帧显式给（上下文帧按真实时间差落位，帧号不连续）。
    frame_pos：[B, f] 或 [f] 的整数（第 i 条样本第 k 帧取 freqs[0][frame_pos[i, k]]）；位置连续时与 causal_rope_apply 逐位相同。
    位置可以为负（见 _rope_time_rows），非负时与原来逐位相同。"""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    frame_pos = torch.as_tensor(frame_pos, dtype=torch.long)
    if frame_pos.dim() == 1:
        frame_pos = frame_pos.unsqueeze(0).expand(x.size(0), -1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        pos = frame_pos[i]          # 留在原设备上做检查，_rope_time_rows 里再搬到频率表的设备
        assert pos.numel() == f, f"frame_pos 第 {i} 条给了 {pos.numel()} 个位置，网格时间维 {f}"
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            _rope_time_rows(freqs[0], pos).view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        # 原来写死 32760（=21x1560）与 x1560，改成在前向里按实际每帧 token 数计算，见 forward 的 KV 分支
        self.max_attention_size = None

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        tf_noisy_start_frame=0,
        tf_ctx_len=None,
        ctx_rope_pos=None,
        rope_offset=0,
        ctx_cache=None,
        ctx_write=False,
        rope_start_frame=None,
        sink_rope_mode=None,
        kv_mem=None,
        kv_capture=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            tf_noisy_start_frame(int): 戊：教师强制时带噪流的 RoPE 起始帧号。ZYT 定长裁剪（去掉 n_0）时为 1，
                干净流仍从 0 起（同一帧号两条流位置编码相同）；默认 0 与原实现逐位一致。
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
            上下文槽参数（全部缺省时与原实现逐位一致）：
            tf_ctx_len(int|None): 教师强制判据改成显式传入（原来靠 s == 2*seq_lens 推断，序列前面多一段上下文时会
                悄悄落到非教师强制分支、后段 token 不加 RoPE）。None 为扩散强制；整数为教师强制，值是序列最前面上下文 token 数，
                序列布局 [上下文 tf_ctx_len | 干净 seq_lens | 带噪 seq_lens]。
            ctx_rope_pos(Tensor [B, K]): 上下文帧的 RoPE 帧位置（context_time_indices 算的，窗口首帧在 rope_offset）。
            rope_offset(int): 干净流、带噪流（以及 KV 推理的全局帧号）整体加的 RoPE 偏移，上下文帧要往前留位置；0 为原实现。
            ctx_cache(dict|None): KV 推理时本层的上下文槽 {"k", "v"}，不参与滚动淘汰、不占 sink 名额；当前帧 query 看 [上下文 | 缓存]。
            ctx_write(bool): 写上下文槽：本次输入整段都是上下文 token，只在它们之间做注意力（与训练掩码一致），K/V 存进 ctx_cache。
            rope_start_frame(int|None): KV 推理时显式指定本次输入的 RoPE 起始帧号（换上下文后重算缓存用），缺省按
                current_start // 每帧 token 数 + rope_offset 算。
            三个参数（只在 KV 推理里起作用，全部缺省时与原实现逐位一致）：
            sink_rope_mode(str|None): None / "absolute" 为原实现（sink 的 K 用写缓存时的绝对位置）；"fixed" 时当前 query 在
                RoPE 坐标 p（本次输入首帧）且 p > local_attn_size=W 时，sink 的 K 从缓存里另存的不带 RoPE 的 k
                （kv_cache["k_sink_raw"]，由调用方预分配，写 sink 槽时原地写入）在位置 p-W+s 现加 RoPE，与 query 的相对距离
                恒为 W-s；p <= W 时 sink 用自己的位置（就是缓存里那份）。窗口帧仍用各自的绝对位置。
                "future" 同 fixed，只是 sink 第 s 帧放在 p+FUTURE_SINK_OFFSET+s（query 前方固定距离，见 sink_policy.py）。
            kv_mem(dict|None): 本层的记忆槽 {"k": 不带 RoPE 的 k [B, cap, H, D], "v": [B, cap, H, D], "n": 有效潜变量数 m}，
                调用方在模型外面填好（预分配张量），模型里只读。第 r 个（r=0 最旧）在 p-W-(m-r) 现加 RoPE；拼成
                [记忆 | sink | 窗口] 只进 K/V、不进 Q。m=0 或 None 时什么都不拼。
            kv_capture(dict|None): 本层的抓取槽 {"k": [B, s, H, D], "v": [B, s, H, D]}（调用方预分配），本次输入的不带 RoPE
                的 k（norm 之后、加 RoPE 之前）与 v 原地拷进去（写缓存那一遍抓记忆库用）。
            注意：FSDP 根单元前向会重建输入的 dict 和 list，模型里对 dict 赋新键回不到调用方，所以
            k_sink_raw、记忆槽、抓取槽都是预分配张量、模型里只做原地写或只读；张量要与 FSDP param_dtype 同为 bf16，
            否则根单元转换输入时会生成拷贝，原地写回不到调用方。
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # 序列并行与上下文槽不能同时启用。
        sp_on = kv_cache is None and not ctx_write and _sp_on()
        if sp_on:
            # 序列并行。本卡 token、全部头 → 全部 token、本卡那组头；之后的 RoPE、补齐、掩码都在完整序列上做
            q, k, v = _sp.seq_to_head(q), _sp.seq_to_head(k), _sp.seq_to_head(v)
            s = q.shape[1]

        if ctx_write:
            # 上下文槽写入。上下文 token 只看上下文（训练掩码里上下文 query 只看有效上下文），时间步 0、带上下文控制，
            # 由调用方保证。K/V（K 已加 RoPE）存进本层 ctx_cache，滑窗缓存一概不动。
            assert ctx_cache is not None and ctx_rope_pos is not None, "写上下文槽要给 ctx_cache 与 ctx_rope_pos"
            roped_query = rope_apply_frames(q, grid_sizes, freqs, ctx_rope_pos).type_as(v)
            roped_key = rope_apply_frames(k, grid_sizes, freqs, ctx_rope_pos).type_as(v)
            ctx_cache["k"] = roped_key
            ctx_cache["v"] = v
            x = attention(roped_query, roped_key, v)
        elif kv_cache is None:
            # if it is teacher forcing training?
            # 不再按长度推断，由 _forward_train 显式传 tf_ctx_len（教师强制为整数，扩散强制为 None）
            is_tf = tf_ctx_len is not None
            if sp_on and not is_tf:
                raise NotImplementedError("序列并行只支持教师强制训练（序列 = 干净流 + 带噪流）")
            if sp_on and (int(tf_ctx_len) or rope_offset):
                raise NotImplementedError("Sequence parallelism with context slots is not supported")
            if is_tf and int(tf_ctx_len) == 0 and not rope_offset:
                # 无上下文槽：与原实现逐句相同
                assert s == seq_lens[0].item() * 2, f"教师强制序列长 {s} 应为两条流 2x{seq_lens[0].item()}"
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    if ii == 1 and tf_noisy_start_frame:
                        rq = causal_rope_apply(q_chunk[ii], grid_sizes, freqs, start_frame=tf_noisy_start_frame).type_as(v)
                        rk = causal_rope_apply(k_chunk[ii], grid_sizes, freqs, start_frame=tf_noisy_start_frame).type_as(v)
                    else:
                        rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                        rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)
            elif is_tf:
                # 三段 [上下文 L | 干净 S | 带噪 S]。上下文按 ctx_rope_pos 逐帧落位；干净流 c_i 在 rope_offset+i，
                # 带噪流 n_j 在 rope_offset+j（裁剪时 tf_noisy_start_frame=1），即原位置整体加 rope_offset，相对位置不变。
                L, S = int(tf_ctx_len), int(seq_lens[0].item())
                assert s == L + 2 * S, f"教师强制序列长 {s} 应为上下文 {L} + 2x{S}"
                parts_q, parts_k = [], []
                if L > 0:
                    fsl = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())
                    assert L % fsl == 0, (L, fsl)
                    ctx_grid = grid_sizes.clone()
                    ctx_grid[:, 0] = L // fsl
                    parts_q.append(rope_apply_frames(q[:, :L], ctx_grid, freqs, ctx_rope_pos).type_as(v))
                    parts_k.append(rope_apply_frames(k[:, :L], ctx_grid, freqs, ctx_rope_pos).type_as(v))
                for a, sf in ((L, rope_offset), (L + S, rope_offset + tf_noisy_start_frame)):
                    parts_q.append(causal_rope_apply(q[:, a:a + S], grid_sizes, freqs, start_frame=sf).type_as(v))
                    parts_k.append(causal_rope_apply(k[:, a:a + S], grid_sizes, freqs, start_frame=sf).type_as(v))
                roped_query = torch.cat(parts_q, dim=1)
                roped_key = torch.cat(parts_k, dim=1)
            if is_tf:
                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :q.shape[1]].transpose(2, 1)  # 原写 :-padded_length，序列长恰为 128 倍数时 padded_length=0、:-0 切成空（704x1280 时 40x880=35200 踩到）

            else:
                # 扩散强制（单流）。教师强制形状的序列若漏传 tf_ctx_len 会走到这里，直接报错而不是静默不加 RoPE
                assert s == seq_lens[0].item(), f"非教师强制分支序列长 {s} 应等于 seq_lens {seq_lens[0].item()}（教师强制要显式传 tf_ctx_len）"
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :q.shape[1]].transpose(2, 1)  # 原写 :-padded_length，序列长恰为 128 倍数时 padded_length=0、:-0 切成空（704x1280 时 40x880=35200 踩到）
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            # 上下文模式下全局帧号整体加 rope_offset；换上下文重算缓存时由 rope_start_frame 显式给真实帧号。两者缺省时与原实现相同
            rope_frame = int(rope_start_frame) if rope_start_frame is not None else current_start_frame + int(rope_offset)
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=rope_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=rope_frame).type_as(v)

            if kv_capture is not None:
                # 抓不带 RoPE 的 k 与 v（原地写进调用方预分配的槽）
                assert tuple(kv_capture["k"].shape) == tuple(k.shape), (kv_capture["k"].shape, k.shape)
                kv_capture["k"].copy_(k)
                kv_capture["v"].copy_(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            if "k_sink_raw" in kv_cache and local_start_index < sink_tokens:
                # sink_rope_mode=fixed 用的 sink 原始 k（不带 RoPE），写 sink 槽时同步原地写（最后一次写缓存为准）
                n_sink_w = min(sink_tokens, local_end_index) - local_start_index
                kv_cache["k_sink_raw"][:, local_start_index:local_start_index + n_sink_w] = k[:, :n_sink_w]
            # 可见窗口 = 全部缓存（无滑窗）或 local_attn_size 帧 x 每帧 token 数
            max_attention_size = local_end_index if self.local_attn_size == -1 \
                else self.local_attn_size * frame_seqlen
            recent_start = local_end_index - (max_attention_size - sink_tokens)
            W_att = self.local_attn_size
            check_sink_rope_mode(sink_rope_mode)
            sink_base = rotated_sink_position(sink_rope_mode, rope_frame, W_att) \
                if W_att != -1 and sink_tokens > 0 else None
            if sink_base is not None:
                # 固定相对距离。sink 第 s 帧在 sink_base+s 现加 RoPE（fixed 是 p-W，future 是 p+FUTURE_SINK_OFFSET，
                # p=rope_frame 是本次 query 的位置）；p > W 时缓存一定已满（或已扩容），可见集合是 sink 加最近 W-S 帧，只换 sink 的 K、V 不变
                assert "k_sink_raw" in kv_cache, "sink_rope_mode=fixed/future 要在 kv_cache 里预分配 k_sink_raw（见 pipeline/lowpoly_rollout.py）"
                assert recent_start >= sink_tokens, (recent_start, sink_tokens)
                sink_grid = grid_sizes.clone()
                sink_grid[:, 0] = self.sink_size
                sink_pos = torch.arange(self.sink_size, dtype=torch.long) + sink_base
                k_sink = rope_apply_frames(kv_cache["k_sink_raw"][:, :sink_tokens], sink_grid, freqs, sink_pos).type_as(v)
                k_att = torch.cat([k_sink, kv_cache["k"][:, recent_start:local_end_index]], dim=1)
                v_att = torch.cat([kv_cache["v"][:, :sink_tokens], kv_cache["v"][:, recent_start:local_end_index]], dim=1)
            elif self.local_attn_size != -1 and sink_tokens > 0 and recent_start > sink_tokens:
                # 缓存容量大于 local_attn_size 时（DMD 长展开的带梯度窗口，见 pipeline/lowpoly_rollout.py
                # 的 _grow_cache），可见集合显式取「sink 加最近 local_attn_size-sink_size 帧」，与容量等于窗口的滚动缓存
                # 看到的集合相同。容量等于 local_attn_size 时 recent_start 不会超过 sink_tokens，走下面原来的分支，逐位不变。
                k_att = torch.cat([kv_cache["k"][:, :sink_tokens], kv_cache["k"][:, recent_start:local_end_index]], dim=1)
                v_att = torch.cat([kv_cache["v"][:, :sink_tokens], kv_cache["v"][:, recent_start:local_end_index]], dim=1)
            else:
                k_att = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
                v_att = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]
            if kv_mem is not None and int(kv_mem["n"]) > 0:
                # 记忆槽（WorldKV 式，免解码）。第 r 个记忆潜变量（r=0 最旧）在 p-W-(m-r) 现加 RoPE，
                # 拼成 [记忆 | sink | 窗口]，只进 K/V
                assert W_att != -1, "记忆槽要配有界缓存（local_attn_size != -1）"
                m_mem = int(kv_mem["n"])
                mem_grid = grid_sizes.clone()
                mem_grid[:, 0] = m_mem
                mem_pos = torch.arange(m_mem, dtype=torch.long) + (rope_frame - W_att - m_mem)
                k_mem = rope_apply_frames(kv_mem["k"][:, :m_mem * frame_seqlen], mem_grid, freqs, mem_pos).type_as(v)
                k_att = torch.cat([k_mem, k_att], dim=1)
                v_att = torch.cat([kv_mem["v"][:, :m_mem * frame_seqlen].to(v_att.dtype), v_att], dim=1)
            if ctx_cache is not None and ctx_cache.get("k") is not None and ctx_cache["k"].shape[1] > 0:
                # 当前帧 query 看 [上下文槽 | sink | 窗口]；上下文槽独立存放，不受滚动淘汰影响
                k_att = torch.cat([ctx_cache["k"].to(k_att.dtype), k_att], dim=1)
                v_att = torch.cat([ctx_cache["v"].to(v_att.dtype), v_att], dim=1)
            x = attention(roped_query, k_att, v_att)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        if sp_on:
            x = _sp.head_to_seq(x)   # 全部 token、本卡那组头 → 本卡 token、全部头
        x = x.flatten(2)
        # x.shape is [1, 65520, 1536]
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        tf_noisy_start_frame=0,
        tf_ctx_len=None,
        ctx_rope_pos=None,
        rope_offset=0,
        ctx_cache=None,
        ctx_write=False,
        rope_start_frame=None,
        sink_rope_mode=None,
        kv_mem=None,
        kv_capture=None
    ):
        r"""
        tf_ctx_len 起的 6 个参数原样透传给自注意力，含义见 CausalWanSelfAttention.forward。
        sink_rope_mode、kv_mem、kv_capture 同样原样透传（缺省 None 与原实现相同）。
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start, tf_noisy_start_frame,
            tf_ctx_len=tf_ctx_len, ctx_rope_pos=ctx_rope_pos, rope_offset=rope_offset,
            ctx_cache=ctx_cache, ctx_write=ctx_write, rope_start_frame=rope_start_frame,
            sink_rope_mode=sink_rope_mode, kv_mem=kv_mem, kv_capture=kv_capture)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        # Wan2.2 TI2V-5B / Fun-5B-Control 的 model_type 是 ti2v（无 CLIP，交叉注意力同 t2v）
        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None
        self._block_mask_key = None  # 掩码按 (类型, 帧数, 每帧 token, 块大小, 滑窗) 缓存，形状一变自动重建

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def set_rope_max_len(self, n: int) -> None:
        """把 RoPE 频率表加长到 n 行（只加长不缩短，默认构造时 1024 行）。rope_params 是逐元素的
        arange(n) 外积再 polar，所以加长后前 1024 行与原表逐位相同（tests/kvmem/test_kvmem_cpu.py 有断言）。
        长推理（05 部署 4000 帧以上，即一千多个潜变量；07 的上下文模式还要加 rope_offset 64）超过 1024 行时要调。"""
        n = int(n)
        if n <= self.freqs.shape[0]:
            return
        d = self.dim // self.num_heads
        dev = self.freqs.device
        self.freqs = torch.cat([
            rope_params(n, d - 4 * (d // 6)),
            rope_params(n, 2 * (d // 6)),
            rope_params(n, 2 * (d // 6))
        ],
            dim=1).to(dev)

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, crop=False
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            # 戊：ZYT 定长裁剪时干净流是 c_0..c_{T-2}、带噪流是 n_1..n_{T-1}，带噪第 j 块对应原第 j+1 块，
            # 可见原「严格更早」的干净块 c_0..c_j，即裁剪后下标 0..j（含 j）
            noise_context_ends[start:end] = (block_index + (1 if crop else 0)) * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                                padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask_ctx(
        device: torch.device | str, num_frames: int, frame_seqlen: int, num_frame_per_block: int,
        crop: bool, num_ctx: int, ctx_valid_n: torch.Tensor
    ) -> BlockMask:
        """带上下文槽的教师强制掩码。序列 [上下文 num_ctx 帧 | 干净 F 帧 | 带噪 F 帧 | 右补齐到 128 倍数]，
        num_ctx 固定（缺的槽补位），长度不随有效个数变，flex_attention 不会按 K 重编译。
        ctx_valid_n：[B] 每条样本的有效上下文帧数，有效槽一律排在前面（数据侧保证）。
        可见性：
          - 有效上下文 query 只看有效上下文（组内双向），不看干净流与带噪流（因果版本不许泄露未来，这是和双向 SFT 唯一不同处）；
          - 干净与带噪 query 看全部有效上下文，再加原 _prepare_teacher_forcing_mask 的可见集合（整体右移 num_ctx 帧）；
          - 补位上下文槽谁都不看它，它自己只看自己（对角，防 NaN）；右补齐 token 同样只看自己。
        num_ctx=0 时可见集合与 _prepare_teacher_forcing_mask 相同。"""
        L = num_ctx * frame_seqlen
        main = num_frames * frame_seqlen
        total_length = L + 2 * main
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        T = total_length + padded_length
        clean_start, clean_end, noisy_end = L, L + main, L + 2 * main
        context_ends = torch.zeros(T, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(T, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(T, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(T, device=device, dtype=torch.long)
        bs = frame_seqlen * num_frame_per_block
        for start in range(clean_start, clean_end, bs):
            context_ends[start:start + bs] = start + bs
        for block_index, start in enumerate(range(clean_end, noisy_end, bs)):
            noise_noise_starts[start:start + bs] = start
            noise_noise_ends[start:start + bs] = start + bs
            # 与原掩码相同：裁剪时带噪第 j 块可见干净下标 0..j，不裁剪时 0..j-1（这里都加上上下文偏移 L）
            noise_context_ends[start:start + bs] = L + (block_index + (1 if crop else 0)) * bs
        vend = (torch.as_tensor(ctx_valid_n, dtype=torch.long).reshape(-1) * frame_seqlen).to(device)
        B = int(vend.numel())

        def attention_mask(b, h, q_idx, kv_idx):
            in_clean_q = (q_idx >= clean_start) & (q_idx < clean_end)
            in_noisy_q = (q_idx >= clean_end) & (q_idx < noisy_end)
            clean_mask = in_clean_q & (kv_idx >= clean_start) & (kv_idx < context_ends[q_idx])
            C1 = (kv_idx >= noise_noise_starts[q_idx]) & (kv_idx < noise_noise_ends[q_idx])
            C2 = (kv_idx >= clean_start) & (kv_idx < noise_context_ends[q_idx])
            noise_mask = in_noisy_q & (C1 | C2)
            kv_valid_ctx = kv_idx < vend[b]
            ctx_mask = kv_valid_ctx & ((q_idx < vend[b]) | in_clean_q | in_noisy_q)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask | ctx_mask

        return create_block_mask(attention_mask, B=B, H=None, Q_LEN=T, KV_LEN=T, _compile=False, device=device)

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        ctx_cache: list = None,
        ctx_write: bool = False,
        ctx_rope_pos=None,
        rope_offset: int = 0,
        rope_start_frame=None,
        sink_rope_mode=None,
        kv_mem: list = None,
        kv_capture: list = None
    ):
        r"""
        Run the diffusion model with kv caching.
        sink_rope_mode（"absolute"/"fixed"）、kv_mem（每层一个记忆槽 dict 的列表）、kv_capture（每层一个抓取槽
        dict 的列表）见 CausalWanSelfAttention.forward；缺省 None 时不放进 kwargs，调用与原实现逐句相同。
        ctx_cache（每层一个 {"k","v"} 的列表）、ctx_write、ctx_rope_pos、rope_offset、rope_start_frame 见
        CausalWanSelfAttention.forward；ctx_write=True 时 x 是上下文潜变量（时间步 0、带上下文控制），只写上下文槽，kv_cache 可为 None。
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # 上下文槽相关参数只在用到时才放进 kwargs，缺省调用与原实现逐句相同
        ctx_kw = {}
        if ctx_cache is not None or ctx_write:
            ctx_kw = {"ctx_write": bool(ctx_write), "ctx_rope_pos": ctx_rope_pos}
        if rope_offset:
            ctx_kw["rope_offset"] = int(rope_offset)
        if rope_start_frame is not None:
            ctx_kw["rope_start_frame"] = int(rope_start_frame)
        if sink_rope_mode is not None and sink_rope_mode != "absolute":
            ctx_kw["sink_rope_mode"] = str(sink_rope_mode)
        kwargs.update(ctx_kw)
        if kv_mem is not None:
            assert len(kv_mem) == len(self.blocks), (len(kv_mem), len(self.blocks))
        if kv_capture is not None:
            assert len(kv_capture) == len(self.blocks), (len(kv_capture), len(self.blocks))

        runtime = getattr(self, "castle_runtime", None)
        if runtime is not None:
            x = runtime.run(self, x, e0, grid_sizes, context, kv_cache,
                            crossattn_cache, current_start, sink_rope_mode=sink_rope_mode,
                            unsupported=(ctx_cache is not None or ctx_write or ctx_rope_pos is not None
                                         or rope_offset != 0 or rope_start_frame is not None
                                         or kv_mem is not None or kv_capture is not None
                                         or cache_start not in (0, None)))
        for block_index, block in (() if runtime is not None else enumerate(self.blocks)):
            if ctx_cache is not None:
                kwargs["ctx_cache"] = ctx_cache[block_index]
            if kv_mem is not None:
                kwargs["kv_mem"] = kv_mem[block_index]          # 只读
            if kv_capture is not None:
                kwargs["kv_capture"] = kv_capture[block_index]  # 原地写
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index] if kv_cache is not None else None,
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index] if kv_cache is not None else None,
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
        tf_crop=False,
        ctx_x=None,
        ctx_valid=None,
        ctx_pos=None,
        rope_offset=None,
    ):
        r"""
        Forward pass through the diffusion model

        上下文槽（只用于教师强制）。ctx_x [B, C_in, K, h, w] 是 K 个上下文潜变量（每个是单帧 VAE 编码）
        拼好控制通道后的模型输入；ctx_valid [B, K] 布尔（有效槽排在前面，补位槽不参与注意力）；ctx_pos [B, K] 上下文的
        RoPE 帧位置；rope_offset 是干净流与带噪流整体加的 RoPE 偏移（缺省 64，即窗口首帧在 64，与双向 SFT 的 time_index 一致）。
        序列变成 [上下文 K | 干净 | 带噪]（tf_crop 之后），上下文时间步 0、不出损失；ctx_x=None 时与原实现逐位一致。

        戊：tf_crop=True（只用于教师强制、逐潜变量）按 ZYT 4.1 固定裁掉 {c_{T-1}, n_0}：
        干净流取第 0..F-2 帧、带噪流取第 1..F-1 帧（时间步与条件同样切），2F 份变 2F-2 份。
        c_{T-1} 从不被任何 token 当作键读、n_0 也不被别的 token 读，所以第 1..F-1 帧的输出与不裁剪时相同。
        返回 F-1 帧（原第 1..F-1 帧），由包装器在前面补第 0 帧。

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        # 原实现只在第一次前向建掩码，帧数、分辨率或 TF/DF 一变就用错掩码；这里按形状做键，变了就重建
        tf_crop = bool(tf_crop) and clean_x is not None
        use_ctx = ctx_x is not None
        if use_ctx:
            assert clean_x is not None, "上下文槽只接了教师强制（要给 clean_x）"
            assert not self.independent_first_frame
        if tf_crop:
            assert self.num_frame_per_block == 1 and not self.independent_first_frame, "定长裁剪只支持逐潜变量教师强制"
            x = x[:, :, 1:]
            clean_x = clean_x[:, :, :-1]
            t = t[:, 1:]
            if aug_t is not None:
                aug_t = aug_t[:, :-1]
        frame_seqlen = x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])
        if use_ctx:
            # 带上下文槽的掩码按（形状、每条样本有效个数）缓存成字典，有效个数只有 0..K 几种，不会每步重建
            n_ctx = ctx_x.shape[2]
            if ctx_valid is None:
                ctx_valid = torch.ones(x.shape[0], n_ctx, dtype=torch.bool, device=device)
            ctx_valid = torch.as_tensor(ctx_valid).to(device=device, dtype=torch.bool)
            n_valid = ctx_valid.long().sum(dim=1)
            assert bool((ctx_valid == (torch.arange(n_ctx, device=device)[None] < n_valid[:, None])).all()), \
                "有效上下文槽必须排在前面"
            ctx_key = ('tfctx', x.shape[2], frame_seqlen, n_ctx, self.num_frame_per_block, tf_crop,
                       tuple(int(v) for v in n_valid.tolist()))
            if getattr(self, "_ctx_block_masks", None) is None:
                self._ctx_block_masks = {}
            if ctx_key not in self._ctx_block_masks:
                self._ctx_block_masks[ctx_key] = self._prepare_teacher_forcing_mask_ctx(
                    device, num_frames=x.shape[2], frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block, crop=tf_crop, num_ctx=n_ctx, ctx_valid_n=n_valid)
            block_mask = self._ctx_block_masks[ctx_key]
        mask_key = ('tf' if clean_x is not None else 'df', x.shape[2], frame_seqlen,
                    self.num_frame_per_block, self.local_attn_size, self.independent_first_frame, tf_crop)
        if not use_ctx and self.block_mask is not None and self._block_mask_key != mask_key:
            self.block_mask = None
        if not use_ctx and self.block_mask is None:   # 无上下文槽时与原实现相同（单槽缓存）
            self._block_mask_key = mask_key
            if clean_x is not None: # TF
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block, crop=tf_crop
                    )
            else: # DF?
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            # clean_x.detach()
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        ctx_len = 0
        if use_ctx:
            # 上下文 token 拼在最前，时间步 0（与 SFT 的上下文帧同为干净、t=0）
            ctx_tok = [self.patch_embedding(u.unsqueeze(0)) for u in ctx_x]
            ctx_tok = torch.cat([u.flatten(2).transpose(1, 2) for u in ctx_tok])
            ctx_len = ctx_tok.shape[1]
            assert ctx_len == n_ctx * frame_seqlen, (ctx_len, n_ctx, frame_seqlen)
            x = torch.cat([ctx_tok.to(x.dtype), x], dim=1)
            t_ctx = torch.zeros(t.shape[0], n_ctx, dtype=t.dtype, device=t.device)
            e_ctx = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_ctx.flatten()).type_as(x))
            e0_ctx = self.time_projection(e_ctx).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t_ctx.shape)
            e0 = torch.cat([e0_ctx, e0], dim=1)
        else:
            block_mask = self.block_mask

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=block_mask)
        if tf_crop:
            kwargs["tf_noisy_start_frame"] = 1  # 戊：带噪流 n_1..n_{T-1} 的 RoPE 帧号从 1 起
        if clean_x is not None:
            kwargs["tf_ctx_len"] = ctx_len     # 教师强制显式告诉注意力（0 = 无上下文，走原实现）
        if use_ctx:
            if ctx_pos is None:
                raise ValueError("给了上下文槽就要给 ctx_pos（RoPE 帧位置）")
            kwargs["ctx_rope_pos"] = torch.as_tensor(ctx_pos).to(device=device, dtype=torch.long)
            kwargs["rope_offset"] = 64 if rope_offset is None else int(rope_offset)

        sp_on = _sp_on()
        if sp_on:
            # 序列并行。token 与逐帧调制 e0 按帧对齐切成 sp 份，本卡只算自己那份
            if clean_x is None:
                raise NotImplementedError("序列并行只支持教师强制训练")
            nf = kwargs["e"].shape[1]
            if nf % _sp.sp_size() or x.shape[1] % nf:
                raise ValueError(f"总帧数 {nf}（token {x.shape[1]}）不能按 sequence_parallel_size={_sp.sp_size()} 按帧均分")
            x = _sp.split_seq(x, dim=1)
            kwargs["e"] = _sp.split_seq(kwargs["e"], dim=1)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if sp_on:
            x = _sp.gather_seq(x)   # 拼回完整序列（反向组内求和）

        if clean_x is not None and use_ctx:
            x = x[:, ctx_len + (x.shape[1] - ctx_len) // 2:]   # 只取带噪流
        elif clean_x is not None:
            x = x[:, x.shape[1] // 2:]
            # [1,32760,1536]
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None or kwargs.get('ctx_write', False):
            return self._forward_inference(*args, **kwargs)
        else:
            # TF or DF
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
