"""Tuning of the RTX 5070 Ti runtime, measured on that GPU at 864 x 480 with a fixed prompt."""

from dataclasses import dataclass, field

# Linear roles of a DiT block. `self_qkv` is one fused projection of the self-attention input.
LINEAR_ROLES = ("self_qkv", "self_o", "cross_q", "cross_o", "ffn_in", "ffn_out")
LINEAR_FORMATS = ("bf16", "fp8", "int8", "mxfp8")
WINDOW_SPLITS = (1, 3, 7, 21)  # equal key ranges of whole frames over the 21-frame attention window


@dataclass(frozen=True)
class RuntimeConfig:
    attention: str = "sage"           # self-attention over the full window: "sage" or "sdpa"
    attention_parts: int = 7          # Sage key ranges; FPS by split: 7: 54.8, 21: 52.7, 3: 49.7, 1: 38.7
    cross_attention: str = "sage"     # cross-attention over the text keys: "sage" or "sdpa"
    # INT8 has about half the error of per-tensor FP8, except on cross_q (outlier channels).
    linears: dict = field(default_factory=lambda: {**dict.fromkeys(LINEAR_ROLES, "int8"), "cross_q": "mxfp8"})

    def __post_init__(self):
        if self.attention not in ("sage", "sdpa") or self.cross_attention not in ("sage", "sdpa"):
            raise ValueError(f"unknown attention backend {self.attention!r} / {self.cross_attention!r}")
        if set(self.linears) != set(LINEAR_ROLES) or not set(self.linears.values()) <= set(LINEAR_FORMATS):
            raise ValueError(f"linears must map {LINEAR_ROLES} to one of {LINEAR_FORMATS}")
        if self.attention_parts not in WINDOW_SPLITS:
            raise ValueError(f"attention_parts must be one of {WINDOW_SPLITS} (whole frames of the 21-frame window)")


PRESETS = {
    "rtx5070ti-sage": RuntimeConfig(attention="sage"),
    "rtx5070ti": RuntimeConfig(attention="sdpa", cross_attention="sdpa"),
}
