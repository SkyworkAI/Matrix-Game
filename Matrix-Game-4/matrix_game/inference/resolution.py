"""Render resolution of the chunked stream: 864x480 by default, or WIDTHxHEIGHT from MATRIX_GAME_RENDER_SIZE.

The Wan2.2 VAE has a 16x spatial stride and the DiT patchifies the latent by 2, so both sides must be multiples of 32. The released
checkpoints were trained at 864x480; any other size is an extrapolation of the model and is not validated.
"""
import os

DEFAULT_SIZE = "864x480"


def render_size():
    text = os.environ.get("MATRIX_GAME_RENDER_SIZE", DEFAULT_SIZE)
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"MATRIX_GAME_RENDER_SIZE must look like 1728x960, got {text!r}") from None
    if width % 32 or height % 32 or width <= 0 or height <= 0:
        raise ValueError(f"MATRIX_GAME_RENDER_SIZE sides must be positive multiples of 32, got {text!r}")
    return width, height


WIDTH, HEIGHT = render_size()
LATENT_WIDTH, LATENT_HEIGHT = WIDTH // 16, HEIGHT // 16
LATENT_SHAPE = (1, 1, 48, LATENT_HEIGHT, LATENT_WIDTH)
