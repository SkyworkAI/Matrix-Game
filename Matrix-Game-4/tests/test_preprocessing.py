import unittest

import numpy as np
from PIL import Image

from matrix_game.inference.preprocessing import crop_and_resize
from matrix_game.inference.offline import frame_blocks


class PreprocessingTests(unittest.TestCase):
    def test_matches_training_torchvision(self):
        from torchvision.transforms import InterpolationMode, functional as F
        rng = np.random.default_rng(0)
        for width, height in ((1280, 720), (1920, 1080), (864, 480), (701, 901), (97, 53)):
            image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
            scale = max(864 / width, 480 / height)
            expected = F.center_crop(F.resize(image, (round(height * scale), round(width * scale)),
                                              interpolation=InterpolationMode.BILINEAR), (480, 864))
            np.testing.assert_array_equal(np.asarray(expected), np.asarray(crop_and_resize(image, 480, 864)))

    def test_offline_padding_does_not_drop_frames(self):
        for size in range(20):
            blocks = list(frame_blocks(iter(range(size))))
            self.assertEqual([v for block, valid in blocks for v in block[:valid]], list(range(size)))
            if blocks:
                self.assertEqual(len(blocks[0][0]), 1)
                self.assertTrue(all(len(block) == 4 for block, _ in blocks[1:]))
                self.assertTrue(all(block[valid:] == [block[valid - 1]] * (len(block) - valid)
                                    for block, valid in blocks))
