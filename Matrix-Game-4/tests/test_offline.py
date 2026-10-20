import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np

from matrix_game.inference import offline


class OfflineTests(unittest.TestCase):
    def test_real_video_io_with_mock_model(self):
        calls = []
        prompt_calls = []
        stream_options = []

        class Components:
            def __init__(self, args):
                self.gen = self.vae = self.prompt = self.sink_latent = None
                self.caption_encoder = lambda value: value
            def load(self): pass
            def config(self): return {"hq_first_frame": False}

        class Stream:
            def __init__(self, *args, **kwargs): stream_options.append(kwargs)
            def set_prompt(self, value): prompt_calls.append(value)
            def push(self, pixels, *, frame_start):
                calls.append((frame_start, len(pixels)))
                return SimpleNamespace(pixels=pixels)
            def close(self): pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, output = root / "input.mp4", root / "output"
            writer = imageio_ffmpeg.write_frames(str(video), (64, 36), fps=48, macro_block_size=1)
            writer.send(None)
            for index in range(10):
                writer.send(np.full((36, 64, 3), index * 20, np.uint8).tobytes())
            writer.close()
            for name in ("models/ancient-egypt/model.safetensors", "models/ancient-egypt/sink.png",
                         "models/racing/model.safetensors", "models/racing/sink.png",
                         "base/models_t5_umt5-xxl-enc-bf16.pth", "base/Wan2.2_VAE.pth"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / "base/google/umt5-xxl").mkdir(parents=True)
            with patch("matrix_game.inference.loader.ModelComponents", Components), \
                    patch("matrix_game.inference.stream.CastleStream", Stream):
                offline.main(["--input", str(video), "--output", str(output), "--resources", str(root)])
            report = json.loads((output / "render.json").read_text())
            self.assertEqual((report["status"], report["frames"], report["padded_frames"], report["fps"]),
                             ("complete", 10, 3, 48))
            self.assertEqual(calls, [(0, 1), (1, 4), (5, 4), (9, 4)])
            self.assertTrue((output / "rendered.mp4").is_file())
            self.assertFalse((output / "rendered.partial.mp4").exists())
            captions = root / "captions.jsonl"
            captions.write_text("\n".join(json.dumps(dict(frame_start=a,frame_end=b,prompt=p))
                for a,b,p in ((0,0,"opening"),(1,4,"turn"),(5,8,"turn"),(9,9,"exit"))))
            calls.clear()
            with patch("matrix_game.inference.loader.ModelComponents", Components), \
                    patch("matrix_game.inference.stream.CastleStream", Stream):
                offline.main(["--scene", "racing", "--input", str(video),
                    "--output", str(root/"racing"), "--resources", str(root),
                    "--prompt-mode", "dynamic", "--captions", str(captions)])
            self.assertEqual(prompt_calls, ["opening", "turn", "exit"])
            self.assertEqual(calls, [(0,1),(1,4),(5,4),(9,4)])
            self.assertEqual(stream_options[-1]["sink_mode"], "first_frame")
            self.assertEqual(stream_options[-1]["sink_rope_mode"], "future")
