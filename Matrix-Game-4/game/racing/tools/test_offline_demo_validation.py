import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("finalize", Path(__file__).with_name("finalize-offline-demo.py"))
finalize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalize)


class GameIntervalTest(unittest.TestCase):
    def setUp(self):
        self.previous = {"simulation_s": 1., "race": {"phase": "racing"}}
        self.row = {"simulation_s": 1. + 3 / 480, "frame": 99,
                    "input_applied": [{}, {}, {}],
                    "race": {"phase": "results", "finished": True}}

    def valid(self):
        return finalize.valid_game_interval(self.previous, self.row, 100, 48, 480)

    def test_finish_within_last_frame(self):
        self.assertTrue(self.valid())

    def test_intermediate_partial_frame_rejected(self):
        self.row["frame"] = 98
        self.assertFalse(self.valid())

    def test_unfinished_partial_frame_rejected(self):
        self.row["race"]["finished"] = False
        self.assertFalse(self.valid())

    def test_step_mismatch_rejected(self):
        self.row["simulation_s"] = 1. + 4 / 480
        self.assertFalse(self.valid())

    def test_normal_frame(self):
        self.row.update(simulation_s=1. + 1 / 48, frame=20)
        self.assertTrue(self.valid())

    def test_pursuit_capture_before_tail(self):
        self.row["frame"] = 40
        self.row["race"]["mode"] = "pursuit"
        self.row["pursuit"] = {"captured": True}
        self.assertTrue(self.valid())

    def test_pursuit_result_freeze(self):
        self.previous["race"]["phase"] = "results"
        self.row.update(simulation_s=1., frame=40, input_applied=[])
        self.row["race"]["mode"] = "pursuit"
        self.row["pursuit"] = {"captured": True}
        self.assertTrue(self.valid())
        self.row["pursuit"]["captured"] = False
        self.assertFalse(self.valid())


if __name__ == "__main__":
    unittest.main()
