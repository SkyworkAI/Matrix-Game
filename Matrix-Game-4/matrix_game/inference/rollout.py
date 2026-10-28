"""Inference schedule and bounded cache initialization extracted from MG4CF."""
from typing import Optional
import torch
from matrix_game.inference.wrappers import init_kv_cache, init_crossattn_cache

def warp_steps(scheduler, steps) -> Optional[torch.Tensor]:
    """官方 warp_denoising_step：1000/750/500/250 -> scheduler.timesteps[1000 - s]（shift 5 网格上 sigma 1、0.9375、0.8333、0.625）。"""
    if steps is None:
        return None
    steps = torch.as_tensor(list(steps), dtype=torch.long)
    timesteps = torch.cat((scheduler.timesteps.cpu().float(), torch.tensor([0.0])))
    out = timesteps[1000 - steps]
    if out[-1] == 0:
        out = out[:-1]
    return out


def set_attention_window(generator, local_attn_size: int = -1, sink_size: int = 0):
    """把因果模型每层自注意力的滑窗与 sink 设成 (W, S)。-1 表示不限（缓存覆盖整段）。
    按模块类型遍历，FSDP 包装与否都能设到真正的注意力模块上。"""
    from matrix_game.inference.wan.modules.causal_model import CausalWanSelfAttention, CausalWanModel
    for mod in generator.modules():
        if isinstance(mod, CausalWanSelfAttention):
            mod.local_attn_size = local_attn_size
            mod.sink_size = sink_size
        elif isinstance(mod, CausalWanModel):
            mod.local_attn_size = local_attn_size


class CausalV2VRollout:
    def __init__(self, generator, scheduler, denoising_step_list: torch.Tensor,
                 denoising_step_list_first: Optional[torch.Tensor] = None, context_noise: int = 0,
                 local_attn_size: int = -1, sink_size: int = 0, cache_write_with_control: bool = False,
                 detach_history_grad: bool = False, num_frame_per_block: int = 1, noise_sync=None,
                 ctx_rope_offset: int = 64, ctx_refresh_recompute: bool = True,
                 sink_rope_mode: str = "absolute", memory_mode: str = "none", memory_max_frames: int = 3,
                 rope_max_len: Optional[int] = None):
        """denoising_step_list / _first：已经 warp 过的时间步（warp_steps 的输出），不含 0。
        cache_write_with_control 只给单元测试造反例用（缓存写入带控制，口径错误），训练与推理一律 False。
        num_frame_per_block 每块潜变量数（默认 1 即逐潜变量）；noise_sync 是可选回调，自展开内部每抽一份
        新噪声（中间步重新加噪、写缓存加 context_noise）就原地调用一次，序列并行时传组内广播，让组内各卡噪声一致。"""
        self.generator = generator
        self.scheduler = scheduler
        self.steps = torch.as_tensor(denoising_step_list).float()
        self.steps_first = None if denoising_step_list_first is None else torch.as_tensor(denoising_step_list_first).float()
        self.context_noise = context_noise
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.cache_write_with_control = cache_write_with_control
        self.detach_history_grad = detach_history_grad  # 默认 False，与官方 CF++ 一致（不截断历史梯度）；True 为截断变体
        self.nfpb = int(num_frame_per_block)
        assert self.nfpb >= 1, num_frame_per_block
        self.noise_sync = noise_sync
        self._model_ref = generator
        self.ctx_rope_offset = int(ctx_rope_offset)          # 只在 run(context_fn=...) 时生效
        self.ctx_refresh_recompute = bool(ctx_refresh_recompute)
        self.ctx_log = []
        # 可选，写缓存那两处前向（首帧写入、每块干净重跑）外面套的上下文（比如 FP8 量化感知训练时
        # 「写缓存那一遍保留 BF16」）。None 时不套，与原来相同。
        self.cache_write_ctx = None
        # 可选，不带梯度的前向用的推理副本（每张卡一份 bf16 完整权重、不经 FSDP，由 trainer 在生成器每次更新后同步）。
        # 设了以后：前缀块、窗口内非退出步、首帧与每块的写缓存、不训练时的自生成首帧都走副本；只有带梯度的前向走 self.generator。
        # 两者权重相同（FSDP 混合精度把 fp32 主权重转 bf16 计算，副本就是同一份 bf16），缓存是外部张量，两边通用。None 时与原来相同。
        self.nograd_generator = None
        # sink 固定相对距离、KV 记忆库、RoPE 表长（见文件头）
        from matrix_game.inference.sink_policy import SINK_ROPE_MODES
        if sink_rope_mode not in SINK_ROPE_MODES:
            raise ValueError(f"sink_rope_mode 只能是 {' / '.join(SINK_ROPE_MODES)}，收到 {sink_rope_mode}")
        if memory_mode not in ("none", "kvbank"):
            raise ValueError(f"memory_mode 只能是 none / kvbank，收到 {memory_mode}")
        self.sink_rope_mode = sink_rope_mode
        self.memory_mode = memory_mode
        self.memory_max_frames = int(memory_max_frames)
        # None：fixed 模式加到 8192 行、future 模式再多 FUTURE_SINK_OFFSET 行，absolute 模式不动（构造时的 1024 行）；整数：至少这么多行（只加长）
        self.rope_max_len = None if rope_max_len in (None, 0) else int(rope_max_len)
        self.mem_log = []


    def _rope_len_target(self):
        if self.rope_max_len is not None:
            return self.rope_max_len
        from matrix_game.inference.sink_policy import rope_positions_needed
        return None if self.sink_rope_mode == "absolute" else rope_positions_needed(self.sink_rope_mode, 8192)


    def _prepare_rope(self):
        """按 rope_max_len 把学生（与推理副本）的 RoPE 频率表加长。不需要时什么都不做。"""
        n = self._rope_len_target()
        if not n:
            return
        from matrix_game.inference.wan.modules.causal_model import CausalWanModel
        for g in (self.generator, self.nograd_generator):
            if g is None:
                continue
            for mod in g.modules():
                if isinstance(mod, CausalWanModel):
                    mod.set_rope_max_len(n)


    def _caches(self, B, num_total_latents, fsl, dtype, device):
        n = num_total_latents if self.local_attn_size == -1 else self.local_attn_size
        kv = init_kv_cache(self.generator, B, n * fsl, dtype, device)
        xc = init_crossattn_cache(self.generator, B, dtype, device)
        return kv, xc


