"""Native packet validation and one-block admission, independent of a game/GPU."""
import queue
import threading
import time
import unittest

import numpy as np

from matrix_game.server.game import CastleGame


def packet(start, count, frame_packets):
    meta = dict(type='frame' if frame_packets else 'chunk', frame_start=start, count=count,
                states=[dict(frame=i, simulation_time=(i + 1) / 40) for i in range(start, start + count)])
    pixels = np.full((count, 720, 1280, 4), start % 256, dtype=np.uint8)
    return meta, pixels.tobytes()


class InputTests(unittest.TestCase):
    def game(self, frames):
        game = CastleGame.__new__(CastleGame)
        game.prepare, game.frame_packets, game.fps, game.received = None, frames, 40, 0
        game.closed = threading.Event()
        game.credit = threading.Semaphore(1)
        game.queue = queue.Queue(maxsize=1)
        game.error = None
        game.ended = None
        return game

    def test_packet_order_and_cadence(self):
        for frames in (False, True):
            game = self.game(frames)
            for start, count in ((0, 1), (1, 4), (5, 4)):
                game.received = start
                packets = iter([packet(i, 1, True) for i in range(start, start + count)] if frames else [packet(start, count, False)])
                game._read = lambda: next(packets)
                rgba, states = game._read_chunk()
                self.assertEqual(rgba.shape, (count, 720, 1280, 4))
                self.assertEqual([state['frame'] for state in states], list(range(start, start + count)))
            for bad in ('order', 'count', 'time', 'payload', 'type'):
                meta, pixels = packet(5, 1 if frames else 4, frames)
                if bad == 'order': meta['frame_start'] = 6
                if bad == 'count': meta['count'] = 2
                if bad == 'time': meta['states'][0]['simulation_time'] = float('nan')
                if bad == 'payload': pixels = pixels[:-1]
                if bad == 'type': meta['type'] = 'invalid'
                game._read = lambda: (meta, pixels)
                with self.assertRaises(ValueError):
                    game._read_chunk()

    def test_bounded_preparation_and_shutdown(self):
        game = self.game(True)
        requests, prepared = [], []
        game.send = lambda value: requests.append(value['frame_start'])
        serial = iter(range(100))
        game._read = lambda: packet(next(serial), 1, True)

        class Prepare:
            def begin(self): pass
            def frame(self, rgba):
                prepared.append(int(rgba[0, 0, 0]))
                return prepared[-1]
            def finish(self, frames): return frames

        game.prepare = Prepare()
        thread = threading.Thread(target=game._produce)
        thread.start()
        try:
            cancel = threading.Event()
            first, _ = game.next(cancel)
            self.assertEqual(first, [0])
            deadline = time.monotonic() + 3
            while game.queue.empty() and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(game.queue.qsize(), 1)
            time.sleep(.03)
            self.assertEqual(requests, [0, 1])
            self.assertEqual(prepared, [0, 1, 2, 3, 4])
            second, _ = game.next(cancel)
            self.assertEqual(second, [1, 2, 3, 4])
            cancel.set()
            self.assertIsNone(game.next(cancel))
        finally:
            game.closed.set()
            game.credit.release()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(game.error)

    def test_terminal_partial_block(self):
        for frames in (False, True):
            game = self.game(frames)
            game.received = 5
            terminal = (dict(type='end', reason='defeated', frame_start=6), b'')
            packets = iter([packet(5, 1, True), terminal] if frames else [terminal])
            game._read = lambda: next(packets)
            self.assertIsNone(game._read_chunk())
            self.assertEqual(game.ended['reason'], 'defeated')
            self.assertIsNone(game.next(threading.Event()))
            game._read = lambda: (dict(type='end', reason='defeated', frame_start=99), b'')
            with self.assertRaises(ValueError):
                game._read_chunk()

    def test_controls_racing_peer_exit(self):
        game = self.game(True)
        game.lock = threading.Lock()
        class Socket:
            def sendall(self, value): raise BrokenPipeError('peer exited')
        game.socket = Socket()
        CastleGame.send(game, dict(type='controls', sequence=1))
        with self.assertRaises(BrokenPipeError):
            CastleGame.send(game, dict(type='request', frame_start=1))
        game.ended = dict(reason='defeated')
        CastleGame.send(game, dict(type='controls', sequence=2))


if __name__ == '__main__':
    unittest.main()
