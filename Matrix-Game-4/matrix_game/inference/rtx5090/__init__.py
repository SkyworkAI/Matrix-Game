"""RTX 5090 inference runtime: fused Triton kernels, FP8 block linears, ring KV window, CUDA graphs.

Self-contained: the reference modules of `matrix_game.inference` are not modified. Selected with
`--preset rtx5090` (or `rtx5090-sage` for SageAttention self-attention, see `rtx5090_sage`).
"""

from .stream import FastStream, load_transformer, warm_up

__all__ = ["FastStream", "load_transformer", "warm_up"]
