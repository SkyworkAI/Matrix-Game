"""Profiling must not alter frames, frame IDs, or publication order."""
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from matrix_game.profiling import ChunkTimer, summarize
from matrix_game.server.app import Session


class ProfileTests(unittest.TestCase):
    def test_timer_and_warmup(self):
        timer = ChunkTimer(True)
        with patch('matrix_game.profiling.time.perf_counter', side_effect=[0., .001, .002, .004]):
            with timer.measure('work_ms'):
                pass
            with timer.measure('work_ms'):
                pass
        self.assertAlmostEqual(timer.values['work_ms'], 3.)
        disabled = ChunkTimer(False)
        with disabled.measure('work_ms'):
            pass
        self.assertEqual(disabled.values, {})
        result = summarize([dict(chunk=i, processing_ms=float(i)) for i in range(30)])
        self.assertEqual(result['chunks'], 5)
        self.assertEqual(result['stages']['processing_ms']['mean'], 27.)
        self.assertEqual(summarize([])['stages'], {})

    def run_session(self, profiling, early, prepared=False):
        dataset = ModuleType('matrix_game.inference.preprocessing')
        dataset.crop_and_resize = lambda image, h, w: image
        dataset.frames_to_tensor = lambda images: np.stack([np.asarray(im) for im in images])
        dataset.tensor_to_uint8 = lambda pixels: pixels
        calls, output = [], []

        class FakeStream:
            def __init__(self, *args, **kwargs):
                self.profile = kwargs['profile']

            def push(self, pixels, frame_start, on_rgb):
                calls.append(frame_start)
                rgb = 255 - pixels
                tensor = SimpleNamespace(numpy=lambda: rgb)
                if on_rgb:
                    on_rgb(tensor)
                return SimpleNamespace(rgb=tensor, pixels=rgb,
                                       timings_ms={'encode': 1.} if self.profile else {})

            def close(self):
                calls.append('closed')

        runtime = SimpleNamespace(args=SimpleNamespace(profile_chunks=profiling, early_publish=early,
            vae_backend='taehv', steps=(1000,), first_steps=(1000,), prompt='test', prompt_mode='fixed'),
            gen=None, vae=None, prompt=None, sink_latent=None, caption_encoder=None,
            stream=lambda **kw: FakeStream(**kw), config=lambda: {})
        session = Session(runtime, 'model', monitor=True)
        data = []
        for start, count in ((0, 1), (1, 4), (5, 4)):
            rgba = np.full((count, 8, 10, 4), start, dtype=np.uint8)
            rgba[..., 3] = 255
            data.append((rgba, [dict(frame=i, simulation_time=i / 30.) for i in range(start, start + count)]))
        iterator = iter(data)
        session.game = SimpleNamespace(next=lambda cancel: next(iterator, None))

        class Output:
            def note_frames(self, count, **kwargs):
                output.append((count, kwargs))

            def __call__(self, packed):
                output.append(packed)

        session.output = Output()
        module = ModuleType('matrix_game.inference.stream')
        module.CastleStream = FakeStream
        with patch.dict('sys.modules', {'matrix_game.inference.preprocessing': dataset, 'matrix_game.inference.stream': module}):
            if prepared:
                from matrix_game.server.reference_input import ReferencePreprocessor
                prep = ReferencePreprocessor(profile=profiling)
                for index, (rgba, states) in enumerate(data):
                    prep.begin()
                    data[index] = (prep.finish([prep.frame(frame) for frame in rgba]), states)
            session.produce()
        self.assertEqual(calls, [0, 1, 5, 'closed'])
        self.assertEqual(session.report['frames'], 9)
        self.assertEqual(session.profile_rows.maxlen, 2048)
        self.assertEqual(len(session.profile_rows), 3 if profiling else 0)
        if profiling:
            for row in session.profile_rows:
                self.assertGreaterEqual(row['cycle_ms'], row['processing_ms'])
                self.assertIn('publish_enqueue_ms', row)
                self.assertEqual(row['cuda_encode_ms'], 1.)
        return output

    def test_frames_unchanged(self):
        baseline = self.run_session(False, False)
        for profile, early in ((True, False), (False, True), (True, True)):
            self.assertEqual(self.run_session(profile, early), baseline)
            self.assertEqual(self.run_session(profile, early, prepared=True), baseline)


if __name__ == '__main__':
    unittest.main()
