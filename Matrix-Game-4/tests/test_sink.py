"""CPU control-flow checks with fake networks; not model-quality validation."""
from pathlib import Path
import sys
import types
import unittest
import tempfile
from unittest.mock import patch

import torch

from matrix_game.inference.stream import CastleStream, encode_sink_image


SHAPE = (1, 1, 48, 30, 54)


class FakeGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def get_scheduler(self):
        return types.SimpleNamespace(add_noise=lambda clean, noise, timestep: clean)

    def forward(self, noisy_image_or_video, conditional_dict, timestep, **kwargs):
        self.calls.append(dict(kwargs, timestep=timestep.clone(),
                               latent=noisy_image_or_video.clone(), cond=conditional_dict))
        pred = conditional_dict.get("control_latent", noisy_image_or_video)
        return None, pred.clone()


class FakeRollout:
    def __init__(self, generator, scheduler, steps, first_steps, **kwargs):
        self.scheduler, self.steps, self.steps_first = scheduler, steps, first_steps
        self.rope_max_len = kwargs["rope_max_len"]

    def _prepare_rope(self):
        pass

    def _caches(self, batch, frames, fsl, dtype, device):
        return [{"k": torch.zeros(batch, frames * fsl, 1, 1)}], [{}]


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.clear_cache()
        self.offline_input = None
        self.raise_on_encode = False

    def clear_cache(self):
        self.encoder_frames = self.decoder_frames = 0
        self.decoded_latents = []

    def encode(self, pixels):
        self.offline_input = pixels.clone()
        self.encoder_frames = 99
        if self.raise_on_encode:
            raise RuntimeError("test encoder failure")
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: torch.ones(1, 48, 1, 30, 54)))

    def encode_to_latent(self, pixels):
        self.offline_input = pixels.clone()
        return torch.ones(SHAPE)

    def encode_stream(self, pixels):
        self.encoder_frames += pixels.shape[2]
        latent = torch.full((1, 48, 1, 30, 54), float(pixels.mean()))
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: latent))

    def decode_stream(self, latent):
        count = 1 if self.decoder_frames == 0 else 4
        self.decoder_frames += count
        self.decoded_latents.append(latent.clone())
        return types.SimpleNamespace(sample=torch.full((1, 3, count, 480, 864), float(latent.mean())))


class SinkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rollout = types.ModuleType("matrix_game.inference.rollout")
        rollout.CausalV2VRollout = FakeRollout
        rollout.set_attention_window = lambda *args: None
        rollout.warp_steps = lambda scheduler, steps: steps
        wrapper = types.ModuleType("matrix_game.inference.wrappers")
        wrapper.frame_seq_length_of = lambda *args: 2
        dataset = types.ModuleType("matrix_game.inference.preprocessing")
        dataset.crop_and_resize = lambda image, h, w: image.resize((w, h))
        dataset.frames_to_tensor = lambda frames: torch.zeros(len(frames), 3, 480, 864)
        self.modules = patch.dict(sys.modules, {
            "matrix_game.inference.rollout": rollout, "matrix_game.inference.wrappers": wrapper,
            "matrix_game.inference.preprocessing": dataset,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.gen, self.vae = FakeGenerator(), FakeVAE()
        self.prompt = torch.zeros(1, 2, 3)

    def stream(self, **kwargs):
        result = CastleStream(self.gen, self.vae, self.prompt, vae_backend="taehv",
                              steps=(1000,), first_steps=(1000, 500), **kwargs)
        self.addCleanup(result.close)
        return result

    def test_external_sink_is_clean_kv_only(self):
        sink = torch.full(SHAPE, .75)
        stream = self.stream(sink_latent=sink, max_latents=3)
        call, = self.gen.calls
        self.assertTrue(call["drop_control"])
        self.assertEqual(call["current_start"], 0)
        self.assertEqual(float(call["timestep"].item()), 0)
        self.assertEqual(call["sink_rope_mode"], "fixed")
        self.assertNotIn("control_latent", call["cond"])
        self.assertTrue(torch.equal(call["latent"], sink))
        self.assertEqual(self.vae.encoder_frames, 0)
        self.assertEqual(self.vae.decoder_frames, 0)
        self.assertEqual(stream.frame, 0)
        frame = 0
        for index in range(3):
            count = 1 if index == 0 else 4
            chunk = stream.push(torch.full((count, 3, 480, 864), .25), frame_start=frame)
            self.assertEqual((chunk.index, chunk.frame_start, chunk.frame_end), (index, frame, frame + count - 1))
            self.assertEqual(chunk.denoising_steps, 2 if index == 0 else 1)
            self.assertTrue(torch.all(chunk.pixels == .25))
            self.assertEqual(self.gen.calls[-1]["current_start"], (index + 1) * stream.fsl)
            frame += count
        self.assertEqual((self.vae.encoder_frames, self.vae.decoder_frames), (9, 9))
        self.assertEqual(sum(c["current_start"] == 0 for c in self.gen.calls), 1)
        with self.assertRaises(ValueError):
            stream.push(torch.zeros(4, 3, 480, 864), frame_start=9)

    def test_no_sink_preserves_bootstrap_positions(self):
        stream = self.stream()
        self.assertEqual(self.gen.calls, [])
        for index, frame in enumerate((0, 1, 5)):
            count = 1 if index == 0 else 4
            chunk = stream.push(torch.zeros(count, 3, 480, 864), frame_start=frame)
            self.assertEqual(chunk.denoising_steps, 2 if index < 2 else 1)
            self.assertEqual(self.gen.calls[-1]["current_start"], index * stream.fsl)

    def test_new_sessions_prefill_new_caches(self):
        sink = torch.ones(SHAPE)
        first = self.stream(sink_latent=sink)
        old_cache = first.kv[0]["k_sink_raw"]
        first.close()
        second = self.stream(sink_latent=sink, max_latents=8192)
        self.assertIsNot(old_cache, second.kv[0]["k_sink_raw"])
        self.assertEqual(second.rollout.rope_max_len, 8193)
        self.assertEqual((second.index, second.frame), (0, 0))

    def test_invalid_sink_fails_before_forward(self):
        for bad in (torch.zeros(1), torch.full(SHAPE, float("nan")), torch.ones(SHAPE, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                self.stream(sink_latent=bad)
        self.assertEqual(self.gen.calls, [])

    def test_image_encoding_and_cache_isolation(self):
        from PIL import Image
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        image = Path(temp.name) / "sink.png"
        Image.new("RGB", (40, 24), (20, 60, 100)).save(image)
        for backend in ("wan", "taehv"):
            latent = encode_sink_image(self.vae, image, backend)
            self.assertEqual(tuple(latent.shape), SHAPE)
            expected = (1, 1, 3, 480, 864) if backend == "wan" else (1, 3, 1, 480, 864)
            self.assertEqual(tuple(self.vae.offline_input.shape), expected)
            self.assertEqual((self.vae.encoder_frames, self.vae.decoder_frames), (0, 0))
        self.vae.raise_on_encode = True
        with self.assertRaisesRegex(RuntimeError, "test encoder failure"):
            encode_sink_image(self.vae, image, "taehv")
        self.assertEqual((self.vae.encoder_frames, self.vae.decoder_frames), (0, 0))


if __name__ == "__main__":
    unittest.main()
