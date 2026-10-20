"""CPU contract tests for release optimizations; no model-quality claims."""
import unittest

import torch

from matrix_game.inference.runtime import plan_cache
from test_sink import SHAPE, SinkTests


class RingTests(unittest.TestCase):
    def test_wrap_rewrite_and_reset(self):
        for window in (2, 5, 21):
            for session in range(2):
                cache = dict(k=torch.zeros(1, window * 2, 1), v=torch.zeros(1, window * 2, 1),
                             global_end_index=torch.zeros(1, dtype=torch.long),
                             local_end_index=torch.zeros(1, dtype=torch.long))
                for position in range(1300):
                    for rewrite in range(3):
                        start, end = plan_cache(cache, position * 2, 2, window, ring=True)
                        cache["k"][:, start:end] = position
                        cache["v"][:, start:end] = position
                        actual = sorted(cache["k"][0, :end:2, 0].tolist())
                        expected = [0] + list(range(max(1, position - window + 2), position + 1))
                        self.assertEqual(actual, expected)
                with self.assertRaises(ValueError):
                    plan_cache(cache, 3000 * 2, 2, window, ring=True)


class PublicationTests(SinkTests):
    def test_publish_before_clean_and_owned_storage(self):
        stream = self.stream(sink_latent=torch.full(SHAPE, .75))
        retained = []
        def receive(rgb):
            self.assertNotEqual(float(self.gen.calls[-1]["timestep"]), 0)
            retained.append(rgb)
        first = stream.push(torch.full((1, 3, 480, 864), .25), frame_start=0, on_rgb=receive)
        snapshot = retained[0].clone()
        second = stream.push(torch.full((4, 3, 480, 864), -.5), frame_start=1, on_rgb=receive)
        self.assertIsNone(first.pixels)
        self.assertEqual(first.rgb.dtype, torch.uint8)
        self.assertEqual(tuple(second.rgb.shape), (4, 480, 864, 3))
        self.assertTrue(torch.equal(retained[0], snapshot))
        self.assertEqual(int(snapshot[0, 0, 0, 0]), 159)
        self.assertEqual(float(self.gen.calls[-1]["timestep"]), 0)

    def test_callback_failure_invalidates_session(self):
        stream = self.stream(sink_latent=torch.full(SHAPE, .75))
        def fail(_):
            raise RuntimeError("broken transport")
        with self.assertRaisesRegex(RuntimeError, "broken transport"):
            stream.push(torch.zeros(1, 3, 480, 864), frame_start=0, on_rgb=fail)
        self.assertTrue(stream.failed)
        with self.assertRaisesRegex(RuntimeError, "closed or failed"):
            stream.push(torch.zeros(1, 3, 480, 864), frame_start=0)


if __name__ == "__main__":
    unittest.main()
