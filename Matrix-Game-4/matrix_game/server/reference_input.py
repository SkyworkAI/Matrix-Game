"""Prepare each arriving game frame with the unchanged training-side pixel math."""
from dataclasses import dataclass

from matrix_game.profiling import ChunkTimer


@dataclass
class PreparedReference:
    images: list
    pixels: object
    timings_ms: dict


class ReferencePreprocessor:
    def __init__(self, *, model=True, profile=False):
        from matrix_game.inference.preprocessing import crop_and_resize, frames_to_tensor
        self.crop_and_resize, self.frames_to_tensor = crop_and_resize, frames_to_tensor
        self.model, self.profile = model, profile

    def begin(self):
        self.timer = ChunkTimer(self.profile)

    def frame(self, rgba):
        from PIL import Image
        with self.timer.measure('rgba_to_rgb_ms'):
            rgb = Image.fromarray(rgba).convert('RGB')
        with self.timer.measure('resize_crop_ms'):
            return self.crop_and_resize(rgb, 480, 864)

    def finish(self, images):
        pixels = None
        if self.model:
            with self.timer.measure('prepare_tensor_ms'):
                pixels = self.frames_to_tensor(images)
        return PreparedReference(images, pixels, self.timer.values)
