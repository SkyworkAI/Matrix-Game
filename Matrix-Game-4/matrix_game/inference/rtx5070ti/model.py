"""Loading and wrapping the causal DiT for the RTX 5070 Ti runtime."""

from functools import partial

import torch

from ..rtx5090 import block as _block, block_graph as _graph
from ..rtx5090.attention import attention
from ..rtx5090.causal_model import FastCausalWanModel
from .config import RuntimeConfig


def _project_qkv(attn, z):
    return attn.qkv.project(z)


class Rtx5070TiCausalWanModel(FastCausalWanModel):
    def __init__(self, model, config: RuntimeConfig):
        super().__init__(model)
        self.config = config
        self._windows = {}   # id(layer K buffer) -> its Sage window
        self._text = {}      # id(layer text K buffer) -> its quantized text keys / values
        self._prompt = None  # the prompt embedding the cached text keys / values belong to
        self._text_buffers = set()  # ids of the text K buffers holding that prompt's keys
        hooks = dict(project_qkv=_project_qkv)
        if config.attention == "sage":
            hooks["self_attention"] = self._self_attention
        if config.cross_attention == "sage":
            hooks["cross_attention"] = self._cross_attention
        self.graphs = _graph.BlockStackGraphs(model.blocks, partial(_block.fused_block, **hooks), compact=True)

    def init_kv_cache(self, batch_size, frame_tokens, device, dtype):
        caches = super().init_kv_cache(batch_size, frame_tokens, device, dtype)
        if self.config.attention == "sage" and any(id(cache["k"]) not in self._windows for cache in caches):
            from .sage import SageWindow
            scratch = {}  # one V range in Sage's layout, shared by every layer's window
            self._windows = {id(cache["k"]): SageWindow(self.config.attention_parts, scratch) for cache in caches}
        return caches

    def _refresh_text_kv(self, context, crossattn_cache) -> None:
        """Computes the text keys / values once, then frees the weights that only serve them (text
        embedding, cross-attention k / v). Later sessions with the same prompt reuse the cached keys
        / values, which stay in the same buffers."""
        if all(cache["is_init"] for cache in crossattn_cache):
            return
        if self._prompt is not None:
            if self._prompt.shape != context.shape or not torch.equal(self._prompt, context):
                raise ValueError("the RTX 5070 Ti runtime supports one fixed prompt per process")
            if {id(cache["k"]) for cache in crossattn_cache} != self._text_buffers:
                # New buffers would need the freed weights to be filled.
                raise ValueError("the text key / value buffers changed after the prompt was encoded")
            for cache in crossattn_cache:
                cache["is_init"] = True
            return
        super()._refresh_text_kv(context, crossattn_cache)
        if self.config.cross_attention == "sage":
            from ..rtx5080.text_attention import SageText
            for cache in crossattn_cache:
                text = self._text[id(cache["k"])] = SageText(cache["k"])
                text.quantize(cache["k"], cache["v"])
        self._prompt = context.detach().clone()
        self._text_buffers = {id(cache["k"]) for cache in crossattn_cache}
        del self.model.text_embedding
        for block in self.model.blocks:
            del block.cross_attn.k, block.cross_attn.v
        torch.cuda.empty_cache()

    def _cross_attention(self, q, cross_k, cross_v):
        return self._text[id(cross_k)].attention(q)

    def _plan_kv(self, kv_cache, current_start, grid):
        plan = super()._plan_kv(kv_cache, current_start, grid)
        for cache in kv_cache if self._windows else ():
            self._windows[id(cache["k"])].prepare(cache["k"], cache["v"], *plan, cache["ring_moved_to"])
        return plan

    def _self_attention(self, q, v, cache_k, cache_v, start, end, window_start, sink_rotated):
        window = self._windows[id(cache_k)]
        if window.usable(cache_k, start, end, window_start):
            return window.attention(q, cache_k, sink_rotated)
        return attention(q, cache_k[:, window_start:end], cache_v[:, window_start:end])
