"""The RTX 5090 causal DiT runtime with SageAttention self-attention over the full window.

Everything except the self-attention call is `rtx5090`: this module only plugs
`sage_attention.SageWindow` into its fused block and keeps the quantised windows up to date.
While the window is still filling, self-attention stays on the dense path of `rtx5090`.
"""

from functools import partial

from ..rtx5090 import attention as _attention, block as _block, block_graph as _graph, causal_model as _base
from .sage_attention import SageWindow


class SageCausalWanModel(_base.FastCausalWanModel):
    """`FastCausalWanModel` whose full-window self-attention runs on SageAttention (INT8 QK, FP8 PV)."""

    def __init__(self, model):
        super().__init__(model)
        self._windows: dict[int, SageWindow] = {}  # id(layer K buffer) -> its quantised companion
        self.graphs = _graph.BlockStackGraphs(
            model.blocks, partial(_block.fused_block, self_attention=self._self_attention))

    def init_kv_cache(self, batch_size, frame_tokens, device, dtype):
        caches = super().init_kv_cache(batch_size, frame_tokens, device, dtype)
        if any(id(cache["k"]) not in self._windows for cache in caches):
            # New K/V buffers (first use or another shape): the base class dropped the graphs too.
            self._windows = {id(cache["k"]): SageWindow() for cache in caches}
        return caches

    def _plan_kv(self, kv_cache, current_start, grid):
        plan = super()._plan_kv(kv_cache, current_start, grid)
        for cache in kv_cache:
            self._windows[id(cache["k"])].prepare(cache["k"], cache["v"], *plan, cache["ring_moved_to"])
        return plan

    def _self_attention(self, q, v, cache_k, cache_v, start, end, window_start, sink_rotated=False):
        window = self._windows[id(cache_k)]
        if window.usable(cache_k, start, end, window_start):
            return window.attention(q, cache_k, v, start, sink_rotated)
        return _attention.attention(q, cache_k[:, window_start:end], cache_v[:, window_start:end])
