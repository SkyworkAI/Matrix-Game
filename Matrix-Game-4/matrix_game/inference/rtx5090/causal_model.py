"""Optimised runtime for the causal DiT: a drop-in for ``CausalWanModel`` in the sampling pipeline.

One call computes the same function as ``CausalWanModel.forward``, organised as

    embed            patch embedding and timestep embedding (eager)
    refresh text     cross-attention K/V of the prompt, only when a layer is marked stale (eager)
    plan KV          where this block's self-attention K/V go, ring eviction (host-side Python)
    block stack      the 30 fused blocks (``block.py``), replayed as one CUDA graph per KV placement
    head             output projection and unpatchify (eager)

Only host-side integers decide the KV placement, and no step reads a value back from the GPU, so
a call returns as soon as its work is queued.
"""

import torch

from .base_model import CausalWanModel, rope_params, sinusoidal_embedding
from .block_graph import BlockStackGraphs, KVPlan, LayerBuffers
from .kernels import rope_rotations


def plan_kv_write(cache: dict, current_start: int, block_tokens: int,
                  sink_tokens: int, window_tokens: int) -> KVPlan:
    """Decide where the block at token offset `current_start` is stored in one layer's K/V buffer.

    Updates the bookkeeping in `cache` (and moves history when the buffer is full) and returns
    the placement the block stack then writes to and attends over. Three cases:

    rewrite   `current_start` is the block written last (another denoising step, or the clean
              pass): same slot, nothing moves.
    append    a new block and the buffer has room: the next free slot, in arrival order.
    evict     a new block and the buffer is full: one old block has to go. The first
              `sink_tokens` (the first frames of the video) are never evicted.

    Eviction as a ring. Shifting the history down by one block would copy almost the whole
    buffer for K and V in every layer. Instead, the newest block always lives in the *last* slot
    and the slots between the sink and the last slot form a ring: before the new block
    overwrites the last slot, the block that was there (now history) is copied over the oldest
    ring slot, and nothing else moves. This is valid because keys are cached *after* RoPE with
    their absolute frame position, so a key means the same thing in whatever slot it is stored,
    and attention is a sum over the window that does not depend on the storage order (only the
    floating-point summation order differs from the shifted layout). Keeping the current block
    in the last slot also keeps the placement constant, so a full window needs one CUDA graph.

    The ring needs whole blocks: if the history between sink and last slot is not a multiple of
    the block size (or the first eviction is partial), the history is shifted instead, exactly
    as in ``CausalSelfAttention.forward``.
    """
    capacity = cache["k"].shape[1]
    global_end, local_end = cache["global_end_index"], cache["local_end_index"]
    current_end = current_start + block_tokens
    if local_end < capacity:
        # A window that is not full (start of a generation) is in arrival order.
        cache["ring_slot"] = None
    cache["ring_moved_to"] = None  # token position an eviction copied the previous block to

    evicted = 0
    if current_end > global_end and local_end + block_tokens > capacity:
        evicted = local_end + block_tokens - capacity
        history = local_end - evicted - sink_tokens  # tokens that stay, excluding the sink
        if history < 0:
            raise ValueError("the KV window cannot hold the sink frames and the current block")
        is_ring = evicted == block_tokens and history % block_tokens == 0 and local_end == capacity
        if is_ring:
            slots = history // block_tokens  # ring length; the last slot is not part of it
            if slots:
                slot = cache["ring_slot"] or 0  # the oldest block in the ring
                target = sink_tokens + slot * block_tokens
                for name in ("k", "v"):
                    buffer = cache[name]
                    buffer.narrow(1, target, block_tokens).copy_(
                        buffer.narrow(1, capacity - block_tokens, block_tokens))
                cache["ring_slot"] = (slot + 1) % slots
                cache["ring_moved_to"] = target
        elif cache["ring_slot"] is not None:
            raise ValueError("the block size changed while the KV window is a ring")
        else:
            for name in ("k", "v"):
                buffer = cache[name]
                buffer[:, sink_tokens:sink_tokens + history] = \
                    buffer[:, sink_tokens + evicted:sink_tokens + evicted + history].clone()

    end = local_end + current_end - global_end - evicted
    start = end - block_tokens
    if start < 0 or end > capacity:
        raise ValueError(f"invalid KV write range [{start}, {end}) for a buffer of {capacity} tokens; "
                         "blocks must be generated in order")
    cache["global_end_index"] = current_end
    cache["local_end_index"] = end
    return KVPlan(start, end, max(0, end - window_tokens))


class FastCausalWanModel:
    """`CausalWanModel` evaluated with fused kernels, a ring KV window and CUDA-graph replay.

    `model` must be loaded, in eval mode, BF16 and on the GPU, with its block linears already
    quantised; its weights must not be moved or replaced afterwards (the graphs hold their
    addresses). One instance serves one generation at a time.
    """

    def __init__(self, model: CausalWanModel):
        weight = model.patch_embedding.weight
        if weight.device.type != "cuda" or weight.dtype != torch.bfloat16:
            raise ValueError("FastCausalWanModel needs a BF16 model on a CUDA device")
        if model.training:
            raise ValueError("FastCausalWanModel needs model.eval()")
        self.model = model
        self.patch_size = model.patch_size
        model.freqs = model.freqs.to(weight.device)
        self.graphs = BlockStackGraphs(model.blocks)
        # K/V buffers, allocated once per shape and handed out again by every init_* call.
        self._kv_buffers: list[tuple[torch.Tensor, ...]] = []  # per layer: k, v, sink keys
        self._kv_signature = None
        self._cross_buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._cross_signature = None

    # ------------------------------------------------------------------ caches
    #
    # The pipeline asks for fresh caches at the start of every generation. A graph is tied to the
    # addresses of the K/V buffers, so the buffers are kept and only the bookkeeping is reset:
    # the returned dicts are new, the tensors inside are the same objects as last time. The graph
    # keys (KV placement and shapes) of the new generation then repeat those of the previous one
    # and every graph is reused without re-capture. Stale contents need no clearing: a block's
    # K/V slot is written before it is read, and attention never reads past the fill level.
    # Asking for another shape allocates new buffers and drops the graphs.

    def init_kv_cache(self, batch_size: int, frame_tokens: int, device, dtype) -> list[dict]:
        """One dict per layer: k, v [B, local_attn_size*frame_tokens, heads, head_dim];
        global_end_index / local_end_index (Python ints, as in `CausalWanModel`);
        ring_slot (None while the window is in arrival order, else the oldest ring slot)."""
        model = self.model
        signature = (batch_size, frame_tokens, torch.device(device), dtype)
        if signature != self._kv_signature:
            self._check_cache_request(device, dtype)
            self.graphs.clear()
            self._kv_buffers = []  # release the old buffers before allocating
            shape = (batch_size, model.local_attn_size * frame_tokens, model.num_heads, model.dim // model.num_heads)
            self._kv_buffers = [(torch.zeros(shape, dtype=dtype, device=device),
                                 torch.zeros(shape, dtype=dtype, device=device),
                                 torch.zeros((batch_size, frame_tokens, model.dim), dtype=dtype, device=device))
                                for _ in range(model.num_layers)]
            self._kv_signature = signature
        return [{"k": k, "v": v, "sink_k": sink_k, "global_end_index": 0, "local_end_index": 0,
                 "ring_slot": None, "ring_moved_to": None}
                for k, v, sink_k in self._kv_buffers]

    def init_crossattn_cache(self, batch_size: int, device, dtype) -> list[dict]:
        """One dict per layer: k, v [B, text_len, heads, head_dim], is_init. Setting is_init=False
        makes the next call recompute the text K/V into the same tensors."""
        model = self.model
        signature = (batch_size, torch.device(device), dtype)
        if signature != self._cross_signature:
            self._check_cache_request(device, dtype)
            self.graphs.clear()
            self._cross_buffers = []
            shape = (batch_size, model.text_len, model.num_heads, model.dim // model.num_heads)
            self._cross_buffers = [(torch.zeros(shape, dtype=dtype, device=device),
                                    torch.zeros(shape, dtype=dtype, device=device))
                                   for _ in range(model.num_layers)]
            self._cross_signature = signature
        return [{"k": k, "v": v, "is_init": False} for k, v in self._cross_buffers]

    def _check_cache_request(self, device, dtype) -> None:
        if dtype != torch.bfloat16 or torch.device(device).type != "cuda":
            raise ValueError("the fused kernels need BF16 K/V caches on the CUDA device")

    def ensure_rope_positions(self, positions: int) -> None:
        """Extend the RoPE table to `positions` frame positions (rows are the same for any length)."""
        model = self.model
        if positions <= model.freqs.shape[0]:
            return
        head_dim = model.dim // model.num_heads
        model.freqs = torch.cat([
            rope_params(positions, head_dim - 4 * (head_dim // 6)),
            rope_params(positions, 2 * (head_dim // 6)),
            rope_params(positions, 2 * (head_dim // 6)),
        ], dim=1).to(model.patch_embedding.weight.device)

    # ------------------------------------------------------------------ forward

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor, y: torch.Tensor,
                 kv_cache: list[dict], crossattn_cache: list[dict], current_start: int,
                 kv_only: bool = False, sink_position: int | None = None) -> torch.Tensor | None:
        """Same contract as `CausalWanModel.forward`: x [B, 48, F, h, w], y [B, 100, F, h, w],
        t [B, F], context [B, 512, 4096] -> flow [B, 48, F, h, w].

        `sink_position`: for a fixed-distance sink (full window only), the frame position its keys are
        rotated to; None keeps the sink at the absolute position it was written at.

        The result is a new tensor on every call (it does not alias a graph buffer).
        """
        model = self.model
        if context.shape[1] != model.text_len:
            raise ValueError(f"context must be padded to {model.text_len} tokens, got {context.shape[1]}")
        if len(kv_cache) != model.num_layers or len(crossattn_cache) != model.num_layers:
            raise ValueError("kv_cache and crossattn_cache need one entry per layer")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, grid = self._embed_patches(x, y)
            e, e0 = self._embed_time(t, x)
            self._refresh_text_kv(context, crossattn_cache)
            plan = self._plan_kv(kv_cache, current_start, grid)

            f, h, w = grid
            rotations = rope_rotations(model.freqs, current_start // (h * w), f, h, w)
            full = plan.end == kv_cache[0]["k"].shape[1]
            if sink_position is not None and not full:
                raise ValueError("a fixed-distance sink needs a full attention window")
            sink_rotations = None if sink_position is None else rope_rotations(model.freqs, sink_position, 1, h, w)
            layers = [LayerBuffers(kv["k"], kv["v"], cross["k"], cross["v"], kv["sink_k"])
                      for kv, cross in zip(kv_cache, crossattn_cache)]
            if kv_only and full:
                # Clean pass over a full window: only the K/V matter, so the last layer stops after
                # writing them and the head is skipped (one extra graph; the cached K/V are the same).
                self.graphs.run(layers, plan, x, e0, rotations, kv_only=True, sink_rotations=sink_rotations)
                return None
            x = self.graphs.run(layers, plan, x, e0, rotations, sink_rotations=sink_rotations)

            # The head reads the graph's output buffer; it is queued before any later replay.
            x = model.head(x, e.unflatten(0, t.shape).unsqueeze(2))
        return self._unpatchify(x, grid)

    def _embed_patches(self, x, y):
        """Latent + control [B, 148, F, h, w] -> tokens [B, F*h'*w', dim] and the token grid."""
        x = self.model.patch_embedding(torch.cat([x, y], dim=1))
        grid = tuple(x.shape[2:])
        return x.flatten(2).transpose(1, 2), grid

    def _embed_time(self, t, x):
        """t [B, F] -> e [B*F, dim] (for the head) and e0 [B, F, 6, dim] (block modulation).

        The sinusoid table is built on the device, so the embedding does not wait for the stream.
        """
        model = self.model
        e = model.time_embedding(sinusoidal_embedding(model.freq_dim, t.flatten()).type_as(x))
        e0 = model.time_projection(e).unflatten(1, (6, model.dim)).unflatten(0, t.shape)
        return e, e0

    def _refresh_text_kv(self, context, crossattn_cache) -> None:
        """Recompute the text keys / values of every layer marked `is_init = False`.

        The values are copied into the existing buffers: the captured graphs read these
        addresses, so a prompt change must not replace the tensors. In the steady state every
        layer is initialised and the text-embedding MLP is skipped altogether.
        """
        if all(cache["is_init"] for cache in crossattn_cache):
            return
        context = self.model.text_embedding(context)
        b = context.shape[0]
        for block, cache in zip(self.model.blocks, crossattn_cache):
            if cache["is_init"]:
                continue
            attn = block.cross_attn
            cache["k"].copy_(attn.norm_k(attn.k(context)).view(b, -1, attn.num_heads, attn.head_dim))
            cache["v"].copy_(attn.v(context).view(b, -1, attn.num_heads, attn.head_dim))
            cache["is_init"] = True

    def _plan_kv(self, kv_cache, current_start, grid) -> KVPlan:
        """Plan the KV write of every layer; all layers advance together and share one placement."""
        model = self.model
        f, h, w = grid
        frame_tokens = h * w
        if current_start % frame_tokens:
            raise ValueError("current_start must be a multiple of the tokens per frame")
        plans = set()
        for cache in kv_cache:
            plans.add(plan_kv_write(cache, current_start, f * frame_tokens,
                                    model.sink_size * frame_tokens, model.local_attn_size * frame_tokens))
        if len(plans) != 1:
            raise ValueError("the per-layer KV caches are out of step with each other")
        return plans.pop()

    def _unpatchify(self, x, grid):
        """[B, F, h*w, pt*ph*pw*C] -> [B, C, F*pt, h*ph, w*pw]."""
        f, h, w = grid
        pt, ph, pw = self.patch_size
        out_dim = self.model.out_dim
        x = x.view(x.shape[0], f, h, w, pt, ph, pw, out_dim)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(x.shape[0], out_dim, f * pt, h * ph, w * pw)
