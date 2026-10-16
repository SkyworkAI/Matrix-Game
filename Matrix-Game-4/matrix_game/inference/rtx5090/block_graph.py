"""CUDA-graph capture / replay of the whole DiT block stack.

One forward of the 30 blocks is a few hundred small launches whose cost is dominated by the
host (Python, dispatch, launch latency). A CUDA graph records the launches once and replays them
with a single call.

What a graph captures
    The launches of ``fused_block`` for every layer, with raw addresses baked in: the model
    weights, the static input buffers (x, e, rotations) owned by the graph entry, each layer's
    self-attention K/V buffers and text K/V buffers, and the *placement* of the block inside the
    K/V buffers (write range and attention window). Values are not captured: a replay reads
    whatever the static inputs, the K/V history and the text K/V hold at that moment, and
    overwrites the current block's K/V slot.

Graph key
    ``(KVPlan(start, end, window_start), x.shape, e.shape, rotations.shape, rotations.dtype)``.
    All layers share one plan. While the window fills there is one graph per fill level; once it
    is full the current block always occupies the last slot (see the ring in ``causal_model.py``), so a
    single graph serves the rest of the video and every later video.

Invalidation
    The graphs are bound to one set of K/V buffers. If a call arrives with different buffer
    objects (the caches were re-allocated, e.g. for another resolution), every graph is dropped
    and captured again on demand. Timestep, latent, control, prompt and history changes never
    invalidate a graph: they only change values in buffers the graph already points at.
    Model weights are assumed immovable: the model is resident on the GPU and is neither offloaded
    nor re-quantised after the runtime is built, so their addresses are not re-checked per call.
"""

from typing import NamedTuple

import torch

from .block import fused_block


class KVPlan(NamedTuple):
    """Placement of the current block in every layer's K/V buffer, in tokens."""
    start: int         # the block's K/V are written to [start, end)
    end: int
    window_start: int  # attention reads [window_start, end)


class LayerBuffers(NamedTuple):
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    cross_k: torch.Tensor
    cross_v: torch.Tensor
    sink_k: torch.Tensor


class _Entry(NamedTuple):
    graph: torch.cuda.CUDAGraph
    x: torch.Tensor
    e: torch.Tensor
    rotations: torch.Tensor
    sink_rotations: torch.Tensor | None
    output: torch.Tensor


def run_block_stack(blocks, layers: list[LayerBuffers], plan: KVPlan,
                    x: torch.Tensor, e: torch.Tensor, rotations: torch.Tensor, block_fn=fused_block,
                    kv_only: bool = False, sink_rotations: torch.Tensor | None = None) -> torch.Tensor:
    """Run all blocks. With `kv_only` the caller only wants the K/V written (the clean pass that
    commits a block to history), so the last block stops once its K/V are in the cache."""
    last = len(blocks) - 1
    for index, (block, layer) in enumerate(zip(blocks, layers)):
        x = block_fn(block, x, e, rotations, layer.cache_k, layer.cache_v,
                     plan.start, plan.end, plan.window_start, layer.cross_k, layer.cross_v,
                     stop_after_kv=kv_only and index == last, sink_k=layer.sink_k, sink_rotations=sink_rotations)
    return x


class BlockStackGraphs:
    """Capture-on-first-use cache of block-stack graphs, keyed as described in the module docstring."""

    def __init__(self, blocks, block_fn=fused_block, compact: bool = False):
        self.blocks = blocks
        self.block_fn = block_fn  # the single-block function every graph of this cache captures
        # `compact`: one capture stream (cuBLAS keeps a workspace per stream) and one memory pool.
        self.compact = compact
        self._pool = self._stream = None
        self._entries: dict[tuple, _Entry] = {}
        self._bound: tuple[torch.Tensor, ...] = ()  # the K/V buffers the captured graphs point at
        self.captures = 0

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        if self._entries:
            # A replay may still be running, and its output may still be read by the eager head.
            torch.cuda.synchronize()
        self._entries.clear()
        self._bound = ()
        self._pool = self._stream = None

    def run(self, layers: list[LayerBuffers], plan: KVPlan,
            x: torch.Tensor, e: torch.Tensor, rotations: torch.Tensor, kv_only: bool = False,
            sink_rotations: torch.Tensor | None = None) -> torch.Tensor:
        """Run the block stack on x [B, S, dim], e [B, F, 6, dim] and the RoPE table.

        Returns the graph's static output buffer [B, S, dim]: it is overwritten by the next replay
        of the same key, so the caller must consume it (launch the head) before calling again.
        """
        buffers = tuple(tensor for layer in layers for tensor in layer)
        if len(buffers) != len(self._bound) or any(a is not b for a, b in zip(buffers, self._bound)):
            self.clear()
            self._bound = buffers

        key = (plan, kv_only, sink_rotations is not None, tuple(x.shape), tuple(e.shape), tuple(rotations.shape), rotations.dtype)
        entry = self._entries.get(key)
        if entry is None:
            entry = self._entries[key] = self._capture(layers, plan, x, e, rotations, kv_only, sink_rotations)
        entry.x.copy_(x)
        entry.e.copy_(e)
        entry.rotations.copy_(rotations)
        if sink_rotations is not None:
            entry.sink_rotations.copy_(sink_rotations)
        entry.graph.replay()
        return entry.output

    def _capture(self, layers, plan, x, e, rotations, kv_only, sink_rotations) -> _Entry:
        # Static inputs are contiguous copies; `copy_` in `run` absorbs any layout of the caller's
        # tensors (x arrives as a transposed view of the patch embedding).
        static_x = x.clone(memory_format=torch.contiguous_format)
        static_e = e.clone(memory_format=torch.contiguous_format)
        static_rotations = rotations.clone(memory_format=torch.contiguous_format)
        static_sink = None if sink_rotations is None else sink_rotations.clone(memory_format=torch.contiguous_format)

        if not self.compact:
            stream = torch.cuda.Stream(device=x.device)
        else:
            if self._stream is None:
                self._stream = torch.cuda.Stream(device=x.device)
            stream = self._stream
        stream.wait_stream(torch.cuda.current_stream(x.device))
        # Run the stack once eagerly on the capture stream first: kernel compilation and library
        # initialisation (cuBLAS handles, attention workspaces) must not happen during capture.
        # Running the stack more than once is harmless: every layer overwrites the current block's
        # K/V slot before reading it, and the history is only read.
        with torch.cuda.stream(stream):
            run_block_stack(self.blocks, layers, plan, static_x, static_e, static_rotations, self.block_fn, kv_only, static_sink)
        torch.cuda.current_stream(x.device).wait_stream(stream)

        # By default each graph gets its own memory pool, so graphs can be replayed in any order.
        # A shared pool (`compact`) lets one replay overwrite another graph's output, which is safe
        # only because each output is consumed (head launched) before the next replay.
        if self.compact and self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool, stream=stream):
            output = run_block_stack(self.blocks, layers, plan, static_x, static_e, static_rotations, self.block_fn, kv_only, static_sink)
        self.captures += 1
        return _Entry(graph, static_x, static_e, static_rotations, static_sink, output)
