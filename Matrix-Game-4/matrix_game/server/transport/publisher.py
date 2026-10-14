#!/usr/bin/env python3
"""GI-only sidecar: framed RGB24 stdin -> loopback RTSP; RTP metadata stdout."""
from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import select
import sys
import threading
import time

from matrix_game.server.transport.output import (frame_bytes, json_bytes, read_packet,
                                        validate_game, write_all)


def compact_game_map(value):
    """Round normalized HUD geometry only, well below a screen pixel."""
    game = value.get('game', {})
    minimap = game.get('map')
    if not isinstance(minimap, dict):
        return value

    def compact(item):
        if isinstance(item, list):
            return [compact(child) for child in item]
        return round(item, 4) if isinstance(item, float) else item

    minimap = dict(minimap)
    for key in ('roads', 'active', 'preview', 'gate'):
        if key in minimap:
            minimap[key] = compact(minimap[key])
    return dict(value, game=dict(game, map=minimap))


class StateWriter:
    """Never block a Gst callback on stdout or a vanished parent."""
    def __init__(self, stream):
        self.stream = stream
        self.pending = deque(maxlen=120)
        self.condition = threading.Condition()
        self.done = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name='rtp-state-output', daemon=True)

    def start(self, frame_pacing):
        write_all(self.stream, json_bytes({'type': 'publisher_ready', 'frame_pacing': frame_pacing}) + b'\n')
        self.thread.start()

    def frame(self, value):
        packet = json_bytes({'type': 'webrtc_frames', 'frames': [compact_game_map(value)]}) + b'\n'
        with self.condition:
            self.pending.append(packet)
            self.condition.notify_all()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Publisher metadata pipe failed') from self.error

    def _run(self):
        try:
            while True:
                with self.condition:
                    while not self.pending and not self.done:
                        self.condition.wait()
                    if not self.pending:
                        break
                    value = self.pending.popleft()
                write_all(self.stream, value)
        except Exception as exc:
            self.error = exc

    def finish(self):
        with self.condition:
            self.done = True
            self.condition.notify_all()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise TimeoutError('Publisher metadata pipe did not drain')
        self.check()
        write_all(self.stream, json_bytes({'type': 'publisher_done'}) + b'\n')


class CheckedInput:
    def __init__(self, fd, check):
        self.fd, self.check = fd, check

    def read(self, size):
        while True:
            self.check()
            if select.select([self.fd], [], [], .1)[0]:
                return os.read(self.fd, size)


class FramePacer:
    """Pace display frames without buffering startup or replaying a stalled schedule."""
    def __init__(self, fps, *, clock=time.monotonic, sleep=time.sleep):
        self.period = 1 / fps
        self.clock, self.sleep = clock, sleep
        self.origin = self.first = None

    def wait(self, index, check):
        now = self.clock()
        if self.origin is None:
            self.origin, self.first = now, index
        deadline = self.origin + (index - self.first) * self.period
        # A stalled producer must not replay its old schedule at full speed.
        if now - deadline > self.period:
            self.origin += now - deadline
            deadline = now
        while now < deadline:
            check()
            self.sleep(min(deadline - now, .025))
            now = self.clock()


def publish(args, publisher_factory=None):
    import numpy as np
    if publisher_factory is None:
        from matrix_game.server.transport.rtsp import RTSPPublisher
        publisher_factory = RTSPPublisher
    size = frame_bytes(args.width, args.height, args.fps)
    writer = StateWriter(sys.stdout.buffer)
    publisher = publisher_factory(args.width, args.height, args.fps, args.generation_id, writer.frame)
    previous, capture = -1, None
    pacer = FramePacer(args.fps) if args.frame_pacing == 'clock' else None

    def check():
        publisher.check()
        writer.check()

    try:
        writer.start(args.frame_pacing)
        source = CheckedInput(sys.stdin.fileno(), check)
        while (packet := read_packet(source, size)) is not None:
            game, payload = packet
            validate_game(game, previous, capture)
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(args.height, args.width, 3)
            if pacer is not None:
                pacer.wait(game['frame'], check)
            publisher.push(rgb, game)
            previous, capture = game['frame'], game['capture_monotonic']
        publisher.finish()
        writer.finish()
    finally:
        publisher.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--fps', type=int, required=True)
    parser.add_argument('--generation-id', required=True)
    parser.add_argument('--frame-pacing', choices=('clock', 'burst'),
                        default=os.getenv('STUDIO_WEBRTC_FRAME_PACING', 'clock'))
    publish(parser.parse_args())


if __name__ == '__main__':
    main()
