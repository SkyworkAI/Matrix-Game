"""Single-GPU inference runtime for absolute and fixed-relative (fixed, future) sinks.

Port of the racing runtime's host KV planning, FP64 RoPE kernel and graph replay.
Only the bounded, chunk1, BF16 inference path is supported; fail closed for
training, retrieved memory, context slots or distributed attention.
"""
from dataclasses import dataclass

import torch


@dataclass
class Plan:
    block: object
    cache: dict
    cross: dict
    start: int
    end: int
    cu_q: torch.Tensor
    cu_self: torch.Tensor
    cu_cross: torch.Tensor


def attention(q, k, v, cu_q, cu_k):
    from matrix_game.inference.wan.modules import attention as attention_module
    if cu_q is None:
        return attention_module.attention(q, k, v)
    flash_attn = attention_module.flash_attn
    b, s = q.shape[:2]
    return flash_attn.flash_attn_varlen_func(
        q=q.flatten(0, 1), k=k.flatten(0, 1), v=v.flatten(0, 1),
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=s, max_seqlen_k=k.shape[1], dropout_p=0.,
        softmax_scale=None, causal=False, window_size=(-1, -1),
        deterministic=False).unflatten(0, (b, s))


def run_blocks(plans, x, e, rotations, sink_rotations, fused=False):
    from matrix_game.inference.rope_kernel import apply_causal_rope
    if fused:
        from matrix_game.inference.kernels import ln_modulate, gate_ln_affine, gate_residual
    for p in plans:
        block, cache = p.block, p.cache
        frames, tokens = e.shape[1], x.shape[1] // e.shape[1]
        mod = (block.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        if fused:
            z, _ = ln_modulate(x, e, block.modulation, block.norm1.eps, 0, 1)
        else:
            z = (block.norm1(x).unflatten(1, (frames, tokens)) * (1 + mod[1]) + mod[0]).flatten(1, 2)
        attn = block.self_attn
        b, s = z.shape[:2]
        q = attn.norm_q(attn.q(z)).view(b, s, attn.num_heads, attn.head_dim)
        k = attn.norm_k(attn.k(z)).view(b, s, attn.num_heads, attn.head_dim)
        v = attn.v(z).view(b, s, attn.num_heads, attn.head_dim)
        q = apply_causal_rope(q, rotations).type_as(v)
        cache["k"][:, p.start:p.end].copy_(apply_causal_rope(k, rotations).type_as(v))
        cache["v"][:, p.start:p.end].copy_(v)
        if p.start == 0:
            cache["k_sink_raw"].copy_(k)
        if sink_rotations is None:
            keys, values = cache["k"][:, :p.end], cache["v"][:, :p.end]
        else:
            sink = apply_causal_rope(cache["k_sink_raw"], sink_rotations).type_as(v)
            keys = torch.cat((sink, cache["k"][:, tokens:p.end]), dim=1)
            values = torch.cat((cache["v"][:, :tokens], cache["v"][:, tokens:p.end]), dim=1)
        y = attn.o(attention(q, keys, values, p.cu_q, p.cu_self).flatten(2))
        if fused:
            normed, x = gate_ln_affine(x, y, e, block.modulation, 2,
                                      block.norm3.weight, block.norm3.bias, block.norm3.eps)
        else:
            x = x + (y.unflatten(1, (frames, tokens)) * mod[2]).flatten(1, 2)
            normed = block.norm3(x)
        attn = block.cross_attn
        q = attn.norm_q(attn.q(normed)).view(b, s, attn.num_heads, attn.head_dim)
        y = attn.o(attention(q, p.cross["k"], p.cross["v"], p.cu_q, p.cu_cross).flatten(2))
        if fused:
            z, x = ln_modulate(x, e, block.modulation, block.norm2.eps, 3, 4, residual=y)
            x = gate_residual(x, block.ffn(z), e, block.modulation, 5)
        else:
            x = x + y
            z = (block.norm2(x).unflatten(1, (frames, tokens)) * (1 + mod[4]) + mod[3]).flatten(1, 2)
            x = x + (block.ffn(z).unflatten(1, (frames, tokens)) * mod[5]).flatten(1, 2)
    return x


def plan_cache(cache, current_start, tokens, window, ring=False):
    """Write current frame in the tail; never evict on denoise/clean rewrites.

    A ring eviction copies only the previous tail into the oldest history slot.
    Absolute RoPE stays attached to each stored key, independent of slot order.
    """
    old_global, old_local = int(cache["global_end_index"]), int(cache["local_end_index"])
    current_end = current_start + tokens
    if current_end not in (old_global, old_global + tokens):
        raise ValueError("Only sequential writes and same-position rewrites are supported")
    evicted = 0
    if current_end > old_global and old_local + tokens > window * tokens:
        evicted = tokens
        for name in ("k", "v"):
            if ring:
                if window > 2:
                    slot = 1 + (current_start // tokens - window) % (window - 2)
                    cache[name][:, slot * tokens:(slot + 1) * tokens].copy_(cache[name][:, -tokens:])
            else:
                cache[name][:, tokens:old_local - tokens].copy_(cache[name][:, 2 * tokens:old_local].clone())
    end = old_local + current_end - old_global - evicted
    if end < tokens or end > window * tokens:
        raise ValueError("Invalid bounded KV range")
    cache["global_end_index"].fill_(current_end)
    cache["local_end_index"].fill_(end)
    return end - tokens, end


class CastleRuntime:
    def __init__(self, mode):
        if mode not in ("kernels", "cudagraph", "ring", "ring-graph", "ring-fused"):
            raise ValueError("Unknown castle runtime")
        self.mode = mode
        self.ring = mode.startswith("ring")
        self.use_graph = mode in ("cudagraph", "ring-graph", "ring-fused")
        self.fused = mode == "ring-fused"
        self.entry = None
        self.captures = self.replays = 0
        self.lengths = {}

    def clear(self):
        if self.entry is not None:
            torch.cuda.synchronize()
        self.entry = None
        self.lengths.clear()

    @torch.no_grad()
    def prime(self, generator, prompt, shape, kv, xc, fsl, sink_rope_mode="fixed"):
        """Capture the steady-window graph before gameplay, then erase warmup KV."""
        if not self.use_graph:
            return
        window = generator.model.blocks[0].self_attn.local_attn_size
        param = next(generator.parameters())
        for cache in kv:
            cache["global_end_index"].fill_((window + 1) * fsl)
            cache["local_end_index"].fill_(window * fsl)
        try:
            generator(noisy_image_or_video=torch.zeros(shape, device=param.device, dtype=param.dtype),
                      conditional_dict={"prompt_embeds": prompt},
                      timestep=torch.zeros(1, 1, device=param.device), drop_control=True,
                      kv_cache=kv, crossattn_cache=xc, current_start=(window + 1) * fsl,
                      sink_rope_mode=sink_rope_mode)
        finally:
            # Text KV contains the real fixed prompt and can be reused. No
            # synthetic history or VAE state may survive into gameplay.
            for cache in kv:
                for name in ("k", "v", "k_sink_raw", "global_end_index", "local_end_index"):
                    cache[name].zero_()

    def run(self, model, x, e, grid_sizes, context, kv, xc, current_start, *,
            sink_rope_mode, unsupported=False):
        from matrix_game.inference.wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
        from matrix_game.inference.wan.modules.causal_model import _sp_on
        from matrix_game.inference.rope_kernel import build_causal_rope_rotations
        from matrix_game.inference.sink_policy import SINK_ROPE_MODES, rotated_sink_position
        if (unsupported or torch.is_grad_enabled() or model.training or _sp_on()
                or sink_rope_mode not in (None,) + SINK_ROPE_MODES or x.dtype != torch.bfloat16
                or not x.is_cuda or FLASH_ATTN_3_AVAILABLE):
            raise ValueError(f"Unsupported runtime: extra={unsupported}, grad={torch.is_grad_enabled()}, "
                             f"training={model.training}, sp={_sp_on()}, sink={sink_rope_mode}, "
                             f"dtype={x.dtype}, device={x.device}, FA2={FLASH_ATTN_2_AVAILABLE}, FA3={FLASH_ATTN_3_AVAILABLE}")
        if grid_sizes.shape != (1, 3) or grid_sizes[0, 0] != 1 or e.shape[1] != 1:
            raise ValueError("Castle runtime requires batch1/chunk1")
        _, h, w = grid_sizes[0].tolist()
        tokens, position = h * w, current_start // (h * w)
        if x.shape[1] != tokens or current_start % tokens or kv is None or xc is None:
            raise ValueError("Expected frame-aligned unpadded input and explicit caches")
        if len(kv) != len(model.blocks) or len(xc) != len(kv):
            raise ValueError("Cache layer count mismatch")
        rotations = build_causal_rope_rotations(model.freqs, 1, h, w, position)
        window = model.blocks[0].self_attn.local_attn_size
        steady = position > window
        sink_position = rotated_sink_position(sink_rope_mode, position, window)
        fixed = sink_position is not None  # the sink's keys are re-rotated (fixed or future policy)
        sink_rotations = build_causal_rope_rotations(model.freqs, 1, h, w, sink_position) if fixed else None

        def cu(n):
            if not FLASH_ATTN_2_AVAILABLE:
                return None
            key = (n, x.device)
            if key not in self.lengths:
                self.lengths[key] = torch.tensor([0, n], dtype=torch.int32, device=x.device)
            return self.lengths[key]

        # Transfer old GPU metadata once, before any block arithmetic starts.
        for cache in kv:
            for name in ("global_end_index", "local_end_index"):
                if cache[name].device.type != "cpu":
                    cache[name] = cache[name].cpu()
        plans = []
        for block, cache, cross in zip(model.blocks, kv, xc):
            a = block.self_attn
            if (a.local_attn_size != window or window < 2 or a.sink_size != 1
                    or cache["k"].shape[1] != window * tokens
                    or cache["k_sink_raw"].shape[1] != tokens
                    or type(block.cross_attn).__name__ != "WanT2VCrossAttention"):
                raise ValueError("Expected identical bounded windows, one sink, and T2V cross attention")
            start, end = plan_cache(cache, current_start, tokens, window, self.ring)
            if start < 0 or end > window * tokens or (fixed and end != window * tokens):
                raise ValueError("Invalid bounded KV range")
            if not cross["is_init"]:
                a = block.cross_attn
                for name, value in (("k", a.norm_k(a.k(context))), ("v", a.v(context))):
                    value = value.view(1, -1, a.num_heads, a.head_dim)
                    if name in cross and cross[name].shape == value.shape:
                        # Graphs hold these addresses. New captions replace values,
                        # never the captured storage or generated self-attention KV.
                        cross[name].copy_(value)
                    else:
                        cross[name] = value
                cross["is_init"] = True
            plans.append(Plan(block, cache, cross, start, end, cu(tokens), cu(end), cu(cross["k"].shape[1])))
        if not self.use_graph or not steady:
            return run_blocks(plans, x, e, rotations, sink_rotations, self.fused)
        return self.graph(model, plans, x, e, rotations, sink_rotations)

    def graph(self, model, plans, x, e, rotations, sink_rotations):
        # One steady-window graph per active session. No growing per-position
        # graph cache: positions and timestep modulation are replay inputs.
        key = (id(model), tuple(x.shape), tuple(e.shape), sink_rotations is not None, tuple(
            (p.start, p.end, *(t.data_ptr() for t in (
                p.cache["k"], p.cache["v"], p.cache["k_sink_raw"],
                p.cross["k"], p.cross["v"]))) for p in plans))
        if self.entry is None or self.entry[0] != key:
            self.clear()
            sx, se, sr = (t.clone() for t in (x, e, rotations))
            ss = None if sink_rotations is None else sink_rotations.clone()
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream())
            # Safe to repeat: every layer overwrites its current KV before read;
            # eviction happened once in the host planner, outside capture.
            with torch.cuda.stream(stream):
                run_blocks(plans, sx, se, sr, ss, self.fused)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                output = run_blocks(plans, sx, se, sr, ss, self.fused)
            self.entry = (key, graph, sx, se, sr, ss, output, plans)
            self.captures += 1
        _, graph, sx, se, sr, ss, output, _ = self.entry
        for dst, src in zip((sx, se, sr, ss), (x, e, rotations, sink_rotations)):
            if dst is not None:
                dst.copy_(src)
        graph.replay()
        self.replays += 1
        return output


def install_runtime(generator, mode):
    if mode not in ("native", "kernels", "cudagraph", "ring", "ring-graph", "ring-fused"):
        raise ValueError("Unknown DiT runtime")
    old = getattr(generator.model, "castle_runtime", None)
    if old is not None:
        old.clear()
    generator.model.castle_runtime = None if mode == "native" else CastleRuntime(mode)
