"""Use Matrix-Game's released loader and streaming inference unchanged."""

from __future__ import annotations

import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NativeSettings:
    source_dir: Path
    arguments: dict


class MatrixGame4Backend:
    def __init__(self, settings: NativeSettings) -> None:
        sys.path.insert(0, str(settings.source_dir))
        from matrix_game.inference.loader import ModelComponents

        self.components = ModelComponents(Namespace(**settings.arguments))
        self.components.load()
        self.stream = None
        self.prompt = self.components.args.prompt

    def start(self, seed: int) -> None:
        self.close()
        self.stream = self.components.stream(seed=seed, max_latents=8192)
        self.prompt = self.components.args.prompt

    def generate(self, frames: np.ndarray, frame_start: int, caption: str | None):
        from matrix_game.inference.preprocessing import (
            frames_to_tensor,
            tensor_to_uint8,
        )

        if caption is not None and caption != self.prompt:
            encoder = self.components.caption_encoder
            if encoder is None:
                raise ValueError(
                    "A changing caption requires the native dynamic caption mode"
                )
            self.stream.set_prompt(encoder(caption))
            self.prompt = caption
        # Preserve the original PIL/uint8 -> float32 operation order and the
        # native clean-KV-before-decode path used by offline inference.
        pixels = frames_to_tensor(frames)
        chunk = self.stream.push(pixels, frame_start=frame_start)
        return (
            tensor_to_uint8(chunk.pixels),
            chunk.index,
            chunk.frame_start,
            chunk.frame_end,
        )

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
