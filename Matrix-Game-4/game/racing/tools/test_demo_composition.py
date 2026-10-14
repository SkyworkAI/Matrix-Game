import importlib.util
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("compose_demo", Path(__file__).with_name("compose-demo.py"))
demo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = demo
spec.loader.exec_module(demo)


class CompositionTests(unittest.TestCase):
    def test_1080p_capture_preserves_legacy_hud_layout(self):
        for mode in ("pursuit", "time_trial"):
            old = {"render_width": 1280, "render_height": 720,
                   "width": 864, "height": 480, "mode": mode}
            new = dict(old, render_width=1920, render_height=1080,
                       hud_layout_width=1280, hud_layout_height=720, hud_render_scale=1.5)
            self.assertEqual(demo.panel_layout(old), demo.panel_layout(new))

    def test_unknown_1080p_hud_layout_is_rejected(self):
        meta = {"render_width": 1920, "render_height": 1080, "width": 864, "height": 480}
        with self.assertRaises(ValueError):
            demo.panel_layout(meta)
        meta.update(hud_layout_width=1280, hud_layout_height=720, hud_render_scale=1)
        with self.assertRaises(ValueError):
            demo.panel_layout(meta)

    def pursuit_fixture(self, **options):
        meta = {"render_width": 1280, "render_height": 720, "width": 864, "height": 480,
                "mode": "pursuit"}
        panels = demo.panel_layout(meta, **options)
        ui = np.zeros((480, 864, 4), np.uint8)
        for panel in panels:
            x1, y1, x2, y2 = panel.box
            ui[y1:y2, x1:x2] = [30, 60, 90, 200]
        return ui, panels

    def test_pursuit_scales_hud_and_map_without_old_panel_remnants(self):
        ui, panels = self.pursuit_fixture()
        self.assertEqual([p.name for p in panels], ["pursuit", "notice", "speed", "map"])
        state = {"race": {"mode": "pursuit", "phase": "racing"}}
        result = demo.reshape_ui(ui, state, panels)
        expected = np.zeros_like(ui)
        for panel in panels:
            x, y = panel.position
            w, h = panel.size
            expected[y:y + h, x:x + w] = [30, 60, 90, 200]
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(panels[-1].size, (180, 180))
        self.assertEqual(panels[-1].position, (670, 286))
        self.assertLess(panels[0].size[0], 180)

    def test_pursuit_map_scale_is_configurable(self):
        _, panels = self.pursuit_fixture(hud_scale=.5, map_size=220)
        self.assertEqual(panels[-1].size, (220, 220))
        self.assertEqual(panels[-1].position, (630, 246))
        self.assertLess(panels[0].size[0], 125)

    def test_capture_tail_retains_compact_layout(self):
        ui, panels = self.pursuit_fixture()
        racing = {"race": {"mode": "pursuit", "phase": "racing"}}
        captured = {"race": {"mode": "pursuit", "phase": "results"},
                    "pursuit": {"captured": True}}
        result = demo.reshape_ui(ui, captured, panels)
        np.testing.assert_array_equal(result, demo.reshape_ui(ui, racing, panels))
        self.assertFalse(np.array_equal(result, ui))

    def test_menus_and_time_trial_results_are_preserved(self):
        ui, panels = self.pursuit_fixture()
        for state in ({"race": {"mode": "pursuit", "phase": "lobby"}},
                      {"race": {"mode": "time_trial", "phase": "results"}}):
            result = demo.reshape_ui(ui, state, panels)
            np.testing.assert_array_equal(result, ui)
            self.assertIsNot(result, ui)

    def test_timestamp_lead_and_release(self):
        events = [{"event": "initial_state", "next_frame": 0, "held_keys": []},
                  {"event": "key", "next_frame": 48, "key": "Up", "down": True},
                  {"event": "key", "next_frame": 60, "key": "Up", "down": False}]
        timeline = demo.InputTimeline(events, 48, 100, 200)
        self.assertEqual(timeline.at(38), set())
        self.assertEqual(timeline.at(39), {"up"})
        self.assertEqual(timeline.at(50), {"up"})
        self.assertEqual(timeline.at(51), set())
        self.assertEqual(timeline.transitions[0]["display_frame"], 39)

    def test_aliases_and_tail_clamp(self):
        events = [{"event": "initial_state", "next_frame": 0, "held_keys": ["W", "F9"]},
                  {"event": "key", "next_frame": 1, "key": "Up", "down": True},
                  {"event": "key", "next_frame": 2, "key": "W", "down": False},
                  {"event": "key", "next_frame": 20, "key": "Up", "down": False}]
        timeline = demo.InputTimeline(events, 48, 20, 200)
        self.assertEqual(timeline.at(0), {"up"})
        self.assertEqual(timeline.at(19), {"up"})

    def test_split_plain_and_translucent_gate(self):
        pixels = np.array([[[230, 177, 37, 230], [23, 23, 12, 253], [6, 9, 9, 235], [231, 236, 232, 255]]], np.uint8)
        clean, gate = demo.split_panel(pixels, (6, 9, 9, 235), np.ones((1, 4), bool), (1, .77, .16))
        np.testing.assert_array_equal(gate[0, 0], pixels[0, 0])
        np.testing.assert_array_equal(clean[0, 0], 0)
        self.assertGreater(gate[0, 1, 3], 180)
        np.testing.assert_array_equal(clean[0, 1], [6, 9, 9, 235])
        np.testing.assert_array_equal(gate[0, 2:], 0)
        np.testing.assert_array_equal(clean[0, 3], pixels[0, 3])

    def test_premultiplied_resize_and_over(self):
        pixels = np.full((2, 2, 4), [50, 25, 0, 128], np.uint8)
        resized = demo.resize_premultiplied(pixels, (4, 4))
        np.testing.assert_array_equal(resized[1, 1], pixels[0, 0])
        base = np.full((4, 4, 4), [100, 100, 100, 255], np.uint8)
        demo.over(base, resized, (0, 0))
        np.testing.assert_array_equal(base[0, 0], [100, 75, 50, 255])

    def test_gate_outside_hud_is_unchanged(self):
        meta = {"render_width": 1280, "render_height": 720, "width": 864, "height": 480}
        ui = np.zeros((480, 864, 4), np.uint8)
        ui[180:220, 430] = [230, 177, 37, 230]
        result = demo.reshape_ui(ui, {"gate": None, "race": {}}, demo.panel_layout(meta))
        np.testing.assert_array_equal(result[180:220, 430], ui[180:220, 430])

    def test_ffmpeg_preserves_overlay_frame_indices(self):
        frames = np.zeros((64, 8, 8, 4), np.uint8)
        frames[..., 3] = 255
        frames[29:, ..., 1] = 200
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "overlay.rgba"
            source.write_bytes(frames.tobytes())
            result = subprocess.check_output([
                "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "8x8",
                "-framerate", "48", "-i", str(source), "-f", "lavfi", "-i", "color=black:s=8x8:r=48",
                "-filter_complex", demo.composition_filter(48), "-map", "[v]", "-frames:v", "64",
                "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
        decoded = np.frombuffer(result, np.uint8).reshape(64, 8, 8, 3)
        self.assertEqual(decoded[28, 0, 0, 1], 0)
        self.assertEqual(decoded[29, 0, 0, 1], 200)


if __name__ == "__main__":
    unittest.main()
