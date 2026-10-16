"""RTX 5070 Ti (16 GB) runtime: the `rtx5090` runtime with INT8 block linears, tuned for 70 SMs.

`--preset rtx5070ti`, or `rtx5070ti-sage` (needs the SageAttention build). Fixed prompts only.
"""

import torch

from ..rtx5080.stream import FastStream, warm_up
from .config import PRESETS, RuntimeConfig


def load_transformer(checkpoint, *, sage: bool = False, config: RuntimeConfig | None = None, device="cuda"):
    """Load the DiT, convert its block linears and wrap it in the runtime (`config` defaults to the preset)."""
    from ..rtx5090.base_model import load_causal_wan
    from .linear import convert_block_linears
    from .model import Rtx5070TiCausalWanModel

    config = config or PRESETS["rtx5070ti-sage" if sage else "rtx5070ti"]
    model = load_causal_wan(str(checkpoint), local_attn_size=21, sink_size=1, device=device, dtype=torch.bfloat16)
    for block in model.blocks:
        convert_block_linears(block, config.linears)
    torch.cuda.empty_cache()
    return Rtx5070TiCausalWanModel(model, config)


__all__ = ["FastStream", "PRESETS", "RuntimeConfig", "load_transformer", "warm_up"]
