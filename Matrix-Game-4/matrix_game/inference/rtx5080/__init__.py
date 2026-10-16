"""RTX 5080 (16 GB) runtime: the `rtx5090` runtime retuned for sm_120 with 84 SMs.

`--preset rtx5080`, or `rtx5080-sage` (needs the SageAttention build). Fixed prompts only.
"""

from .model import load_transformer
from .stream import FastStream, warm_up

__all__ = ["FastStream", "load_transformer", "warm_up"]
