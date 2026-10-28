import unittest

from matrix_game.inference.sink_policy import (FUTURE_SINK_OFFSET, SINK_ROPE_MODES, rope_positions_needed,
                                               rotated_sink_position)


class SinkPolicyTests(unittest.TestCase):
    def test_positions(self):
        window = 21
        for position in (0, 1, window):
            for mode in SINK_ROPE_MODES:
                self.assertIsNone(rotated_sink_position(mode, position, window))
        for position in (window + 1, 1000, 8191):
            self.assertIsNone(rotated_sink_position("absolute", position, window))
            self.assertEqual(rotated_sink_position("fixed", position, window), position - window)
            self.assertEqual(rotated_sink_position("future", position, window), position + FUTURE_SINK_OFFSET)
        self.assertEqual(FUTURE_SINK_OFFSET, 11)

    def test_rope_rows(self):
        self.assertEqual(rope_positions_needed("absolute", 8192), 8192)
        self.assertEqual(rope_positions_needed("fixed", 8192), 8192)
        self.assertEqual(rope_positions_needed("future", 8192), 8192 + FUTURE_SINK_OFFSET)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            rotated_sink_position("dual", 30, 21)


if __name__ == "__main__":
    unittest.main()
