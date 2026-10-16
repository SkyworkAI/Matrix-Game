"""Loading and wrapping the causal DiT for the RTX 5080 runtime."""

from functools import partial

import torch

from ..rtx5090 import block as _block, block_graph as _graph
from ..rtx5090.causal_model import FastCausalWanModel
from .linears import project_qkv, quantize_blocks


class _Rtx5080Graphs:
    """Block-stack graphs with this runtime's GEMMs and optional SageAttention cross-attention."""

    def _use_graphs(self, model, sage_text: bool, **hooks):
        self._text = {} if sage_text else None  # id(layer text K buffer) -> SageText
        if sage_text:
            hooks["cross_attention"] = self._cross_attention
        self.graphs = _graph.BlockStackGraphs(
            model.blocks, partial(_block.fused_block, project_qkv=project_qkv, **hooks), compact=True)

    def _refresh_text_kv(self, context, crossattn_cache) -> None:
        stale = [cache for cache in crossattn_cache if not cache["is_init"]]
        super()._refresh_text_kv(context, crossattn_cache)
        if self._text is None or not stale:
            return
        from .text_attention import SageText
        current = {id(cache["k"]) for cache in crossattn_cache}
        self._text = {key: value for key, value in self._text.items() if key in current}
        for cache in stale:
            window = self._text.get(id(cache["k"]))
            if window is None:
                window = self._text[id(cache["k"])] = SageText(cache["k"])
            window.quantize(cache["k"], cache["v"])

    def _cross_attention(self, q, cross_k, cross_v):
        return self._text[id(cross_k)].attention(q)


class Rtx5080CausalWanModel(_Rtx5080Graphs, FastCausalWanModel):
    """`FastCausalWanModel` with the 16-row-aligned / transposed FP8 GEMMs and compact graphs."""

    def __init__(self, model):
        super().__init__(model)
        self._use_graphs(model, sage_text=False)


def _sage_model(model):
    """SageAttention for the full-window self-attention (``self_attention``) and the cross-attention."""
    from ..rtx5090_sage.causal_model import SageCausalWanModel

    from .self_attention import Rtx5080SageWindow

    class Rtx5080SageCausalWanModel(_Rtx5080Graphs, SageCausalWanModel):
        def __init__(self, model):
            super().__init__(model)
            self._use_graphs(model, sage_text=True, self_attention=self._self_attention)

        def init_kv_cache(self, batch_size, frame_tokens, device, dtype):
            caches = FastCausalWanModel.init_kv_cache(self, batch_size, frame_tokens, device, dtype)
            if any(id(cache["k"]) not in self._windows for cache in caches):
                # New K/V buffers (first use or another shape): the base class dropped the graphs too.
                self._windows = {id(cache["k"]): Rtx5080SageWindow() for cache in caches}
            return caches

    return Rtx5080SageCausalWanModel(model)


def load_transformer(checkpoint, *, sage: bool = False, device="cuda"):
    """Load the DiT and quantise it block by block on the CPU (GPU load peak ~5 GB instead of ~10)."""
    from ..rtx5090.base_model import load_causal_wan

    model = load_causal_wan(str(checkpoint), local_attn_size=21, sink_size=1, device="cpu", dtype=torch.bfloat16)
    quantize_blocks(model, device)
    model.to(device)
    return _sage_model(model) if sage else Rtx5080CausalWanModel(model)
