import argparse
import unittest
from pathlib import Path

import torch

from matrix_game.scenes import add_model_arguments, resolve_model_arguments
from matrix_game.server.racing import controls, game_options
from matrix_game.inference.seq_parallel import init_sequence_parallel, sp_size
from test_sink import SinkTests, SHAPE


class RacingStreamTests(SinkTests):
    def test_first_frame_is_displayed_and_uses_separate_encoder_history(self):
        sink = torch.full(SHAPE, .75)
        stream = self.stream(sink_latent=sink, sink_mode="first_frame", sink_rope_mode="future")
        self.assertEqual(self.gen.calls, [])
        first = stream.push(torch.full((1, 3, 480, 864), .25), frame_start=0)
        self.assertEqual(first.denoising_steps, 0)
        self.assertTrue(torch.all(first.pixels == .75))
        self.assertEqual(len(self.gen.calls), 1)
        self.assertEqual(float(self.gen.calls[0]["timestep"]), 0)
        self.assertEqual(self.gen.calls[0]["sink_rope_mode"], "future")
        self.assertTrue(torch.all(first.control == .25))
        next_chunk = stream.push(torch.zeros(4, 3, 480, 864), frame_start=1)
        self.assertEqual(next_chunk.denoising_steps, 1)
        self.assertEqual(self.gen.calls[-1]["current_start"], stream.fsl)
        self.assertEqual((self.vae.encoder_frames, self.vae.decoder_frames), (5, 5))

    def test_prompt_update_preserves_self_kv_and_frame_count(self):
        stream = self.stream(sink_latent=torch.full(SHAPE, .75), sink_mode="first_frame", sink_rope_mode="future")
        cache = stream.kv[0]
        stream.xc[0]["is_init"] = True
        new = self.prompt + 1
        stream.set_prompt(new)
        self.assertIs(stream.kv[0], cache)
        self.assertFalse(stream.xc[0]["is_init"])
        self.assertEqual(stream.frame, 0)
        stream.push(torch.zeros(1, 3, 480, 864), frame_start=0)
        self.assertIs(self.gen.calls[-1]["cond"]["prompt_embeds"], new)


class RacingOptionsTests(unittest.TestCase):
    def test_profile(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        args = resolve_model_arguments(parser.parse_args(["--scene", "racing", "--prompt-mode", "dynamic"]), False)
        self.assertEqual((args.sink_mode, args.sink_rope_mode, args.vae_backend), ("first_frame", "future", "taehv"))
        self.assertEqual(args.steps, (1000,))
        self.assertEqual(args.prompt_interval_chunks, 1)
        self.assertEqual(args.asset_root.name, "racing")

    def test_sink_rope_mode_flag(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        for argv, mode in ((["--scene", "racing"], "future"),
                           (["--scene", "racing", "--sink-rope-mode", "absolute"], "absolute"),
                           (["--scene", "castle"], "fixed"),
                           (["--scene", "castle", "--sink-rope-mode", "future"], "future")):
            self.assertEqual(resolve_model_arguments(parser.parse_args(argv), False).sink_rope_mode, mode)

    def test_inputs(self):
        self.assertEqual(game_options({"mode": "pursuit"})["route"], "avenue")
        self.assertEqual(controls(dict(sequence=1, drive=[1, -.5, 0, 0]))["drive"], [1, -.5, 0, 0])
        for value in ({"mode": "unknown"}, {"front_drive": 1.01}, {"front_grip": float("nan")}, {"path": "x"}):
            with self.assertRaises(ValueError):
                game_options(value)
        for value in (dict(sequence=1, drive=[1, 0, -1, 0]), dict(sequence=True), dict(sequence=1, action="exec")):
            with self.assertRaises(ValueError):
                controls(value)

    def test_no_distributed_launch(self):
        self.assertEqual(sp_size(), 1)
        init_sequence_parallel(1)
        with self.assertRaises(ValueError):
            init_sequence_parallel(2)


if __name__ == "__main__":
    unittest.main()
