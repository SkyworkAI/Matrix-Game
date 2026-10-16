"""SageAttention self-attention window with the split measured on the RTX 5080.

Attention + two range requantisations per forward, by number of equal key ranges (us):
3: 307, 5: 259, 7: 241, 9: 252, 15: 257, 21: 288, 27: 299, 35: 341, 45: 387.
"""

from ..rtx5090_sage.sage_attention import SageWindow

PARTS = 7


class Rtx5080SageWindow(SageWindow):
    def __init__(self, parts: int = PARTS):
        super().__init__(parts)
