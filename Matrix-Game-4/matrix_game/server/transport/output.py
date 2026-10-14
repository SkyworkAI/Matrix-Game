"""Bounded model RGB transport to an isolated GStreamer Python interpreter.

Only decoded display frames may be evicted. Input reads, inference, and game
credits remain owned by CausalEngine. Neither pipe IO nor HTTP consumption runs
on the inference thread. This module deliberately has no ML or GI imports.
"""
from __future__ import annotations

import asyncio
from collections import deque
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import threading
import time

MAX_GAME_FRAME_BYTES = 64 * 1024


class Stopped(Exception):
    """The display publisher was cancelled by its owning session."""


STATE_MIME = 'application/x-ndjson'
HEADER = struct.Struct('>II')  # JSON bytes, RGB24 bytes; no pickle on either pipe.
MAX_RECORD_BYTES = 1024 * 1024
MAX_RGB_BYTES = 64 * 1024 * 1024


def frame_bytes(width, height, fps):
    if any(type(value) is not int or value <= 0 for value in (width, height, fps)):
        raise ValueError('WebRTC dimensions and integer FPS must be positive')
    size = width * height * 3
    if size > MAX_RGB_BYTES:
        raise ValueError('WebRTC RGB frame exceeds size limit')
    return size


def validate_game(game, previous_frame=-1, previous_capture=None):
    if not isinstance(game, dict) or type(game.get('frame')) is not int:
        raise ValueError('WebRTC requires an integer game frame index')
    if game['frame'] <= previous_frame:
        raise ValueError('Game frame indices must be strictly increasing')
    capture = game.get('capture_monotonic')
    if (type(capture) not in (int, float) or not math.isfinite(capture)
            or capture < 0 or (previous_capture is not None and capture < previous_capture)):
        raise ValueError('Game capture timestamps must be finite and nondecreasing')


def json_bytes(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')


def write_all(stream, value):
    remaining = memoryview(value)
    while remaining:
        count = stream.write(remaining)
        if not count:
            raise BrokenPipeError('Publisher pipe stopped accepting bytes')
        remaining = remaining[count:]


def read_exact(stream, size, *, eof_ok=False):
    parts = bytearray()
    while len(parts) < size:
        value = stream.read(size - len(parts))
        if not value:
            if eof_ok and not parts:
                return None
            raise ValueError('Truncated WebRTC RGB packet')
        parts.extend(value)
    return bytes(parts)


def read_packet(stream, expected_rgb_bytes):
    header = read_exact(stream, HEADER.size, eof_ok=True)
    if header is None:
        return None
    metadata_size, rgb_size = HEADER.unpack(header)
    if not 0 < metadata_size <= MAX_GAME_FRAME_BYTES or rgb_size != expected_rgb_bytes:
        raise ValueError('Invalid WebRTC RGB packet lengths')
    metadata = json.loads(read_exact(stream, metadata_size))
    if not isinstance(metadata, dict) or set(metadata) != {'game'}:
        raise ValueError('Invalid WebRTC RGB packet metadata')
    validate_game(metadata['game'])
    return metadata['game'], read_exact(stream, rgb_size)


def sidecar_environment():
    env = os.environ.copy()
    # The GI interpreter can have a different Python ABI from the model venv.
    # The package is resolved from cwd; never import the model's binary modules.
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


class WebRTCModelOutput:
    """Inference-thread emitter and independent bounded child-pipe workers."""

    def __init__(self, width, height, fps, generation_id, cancel, report,
                 *, queue_size=8, metadata_size=120, finish_timeout=10):
        self.rgb_size = frame_bytes(width, height, fps)
        if not re.fullmatch(r'[a-f0-9]{32}', generation_id):
            raise ValueError('Invalid WebRTC generation ID')
        if queue_size < 1 or metadata_size < 1 or finish_timeout <= 0:
            raise ValueError('WebRTC queue sizes and finish timeout must be positive')
        python = os.environ.get('STUDIO_WEBRTC_PYTHON', '').strip()
        if not python:
            raise ValueError('STUDIO_WEBRTC_PYTHON must name the GI-enabled Python interpreter')
        root = Path(__file__).resolve().parents[3]
        self.command = [python, '-u', '-m', 'matrix_game.server.transport.publisher',
                        '--width', str(width), '--height', str(height), '--fps', str(fps),
                        '--generation-id', generation_id]
        self.root = root
        self.cancel, self.report = cancel, report
        self.finish_timeout = finish_timeout
        self.lock = threading.Condition()
        self.frames = deque(maxlen=queue_size)
        self.metadata = deque(maxlen=metadata_size)
        self.declared = deque()
        self.next_frame = 0
        self.last_capture = None
        self.last_chunk = None
        self.last_encoded_frame = -1
        self.process = None
        self.threads = []
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.input_closed = threading.Event()
        self.closing = threading.Event()
        self.input_done = False
        self.terminal = False
        self.error = None
        self.stderr = deque(maxlen=32)
        self.close_lock = threading.Lock()
        self.closed = False
        report.update(media_encoder='gstreamer-x264', media_encoded_frames=0,
                      media_dropped_frames=0, metadata_evicted=0,
                      media_queue_capacity=queue_size, media_queue_depth=0)

    def check(self):
        with self.lock:
            error = self.error
        if error is not None:
            raise error

    def _fail(self, error):
        with self.lock:
            if self.error is None and not self.closing.is_set():
                self.error = error
            self.cancel.set()
            self.lock.notify_all()
        self._terminate()

    def _terminate(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass

    async def start(self):
        self.process = subprocess.Popen(self.command, cwd=self.root, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                        env=sidecar_environment())
        for name, target in [('rgb', self._write), ('state', self._read),
                             ('stderr', self._read_stderr), ('watch', self._watch)]:
            thread = threading.Thread(target=target, name='webrtc-' + name, daemon=True)
            self.threads.append(thread)
            thread.start()
        deadline = time.monotonic() + self.finish_timeout
        while not self.ready.is_set():
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError('WebRTC publisher did not become ready')
            await asyncio.sleep(.01)
        self.check()

    def note_frames(self, count, ready_wall_seconds=None, chunk_index=None, *, game_frames=None):
        self.check()
        ready = time.time() if ready_wall_seconds is None else ready_wall_seconds
        if type(count) is not int or not 1 <= count <= 256 or not math.isfinite(ready):
            raise ValueError('Invalid WebRTC RGB group count or ready time')
        if self.declared:
            raise ValueError('Previous declared game frames have no RGB payload')
        if (type(chunk_index) is not int or not 0 <= chunk_index <= 0xFFFFFFFF
                or (self.last_chunk is not None and chunk_index < self.last_chunk)):
            raise ValueError('WebRTC model chunk indices must be nondecreasing uint32 values')
        if game_frames is None or len(game_frames) != count:
            raise ValueError('WebRTC requires exact game_frames for every decoded RGB frame')
        capture = self.last_capture
        declared = []
        for offset, game in enumerate(game_frames):
            validate_game(game, self.next_frame + offset - 1, capture)
            if game['frame'] != self.next_frame + offset:
                raise ValueError('Decoded RGB cadence does not match game frame indices')
            capture = game['capture_monotonic']
            # Freeze state before another input read can replace/mutate metadata.
            payload = json_bytes({'game': game})
            if len(payload) > MAX_GAME_FRAME_BYTES:
                raise ValueError('Game presentation metadata exceeds the per-frame limit')
            declared.append(payload)
        self.declared.extend(declared)
        self.last_capture, self.last_chunk = capture, chunk_index

    def __call__(self, payload):
        self.check()
        if not isinstance(payload, bytes):
            raise ValueError('WebRTC emitter requires packed RGB24 bytes')
        count, remainder = divmod(len(payload), self.rgb_size)
        if remainder or not count or count > len(self.declared):
            raise ValueError('RGB24 payload does not match declared game frames')
        for offset in range(0, len(payload), self.rgb_size):
            packet = (self.declared.popleft(), payload[offset:offset + self.rgb_size])
            with self.lock:
                if self.cancel.is_set() or self.closing.is_set():
                    self.check()
                    raise Stopped()
                if self.input_done:
                    raise RuntimeError('RGB submitted after publisher input completed')
                if len(self.frames) == self.frames.maxlen:
                    self.report['media_dropped_frames'] += 1
                self.frames.append(packet)
                self.next_frame += 1
                self.report['media_queue_depth'] = len(self.frames)
                self.lock.notify_all()

    def inference_done(self, future):
        """Executor completion drives encoder EOF even when HTTP is not consumed."""
        if self.closing.is_set() or self.cancel.is_set():
            self._terminate()
            return
        if future.cancelled():
            self._fail(RuntimeError('Model inference was cancelled'))
            return
        error = future.exception()
        if error is not None:
            self._fail(error)
            return
        if self.declared:
            self._fail(ValueError('Model finished before every declared game frame had RGB'))
            return
        with self.lock:
            self.input_done = True
            self.lock.notify_all()

    def _write(self):
        try:
            while not self.cancel.is_set() and not self.closing.is_set():
                with self.lock:
                    while not self.frames and not self.input_done and not self.cancel.is_set():
                        self.lock.wait(.1)
                    if self.cancel.is_set() or self.closing.is_set():
                        return
                    if not self.frames:
                        break
                    metadata, rgb = self.frames.popleft()
                    self.report['media_queue_depth'] = len(self.frames)
                for value in (HEADER.pack(len(metadata), len(rgb)), metadata, rgb):
                    write_all(self.process.stdin, value)
        except Exception as exc:
            if not self.closing.is_set() and not self.cancel.is_set():
                self._fail(RuntimeError(f'WebRTC publisher input failed: {exc}'))
        finally:
            self.process.stdin.close()
            self.input_closed.set()

    def _read_stderr(self):
        while value := self.process.stderr.read(2048):
            with self.lock:
                self.stderr.append(value.decode(errors='replace'))

    def _read(self):
        pending = bytearray()
        try:
            while value := self.process.stdout.read(64 * 1024):
                pending.extend(value)
                while b'\n' in pending:
                    end = pending.index(b'\n')
                    if end > MAX_RECORD_BYTES:
                        raise ValueError('Publisher state record exceeds limit')
                    record = bytes(pending[:end])
                    del pending[:end + 1]
                    self._record(json.loads(record), len(record) + 1)
                if len(pending) > MAX_RECORD_BYTES:
                    raise ValueError('Publisher state record exceeds limit')
            if pending:
                raise ValueError('Truncated publisher state record')
            if not self.closing.is_set() and not self.cancel.is_set():
                code = self.process.wait(timeout=2)
                if code or not self.terminal:
                    with self.lock:
                        detail = ''.join(self.stderr)[-4000:]
                    raise RuntimeError(f'WebRTC publisher exited without clean EOF ({code}): {detail}')
        except Exception as exc:
            if not self.closing.is_set() and not self.cancel.is_set():
                self._fail(exc)
        finally:
            self.finished.set()

    def _record(self, record, size):
        if not isinstance(record, dict) or self.terminal:
            raise ValueError('Unexpected publisher record')
        kind = record.get('type')
        if kind == 'publisher_ready' and not self.ready.is_set():
            pacing = record.get('frame_pacing', 'burst')
            if pacing not in ('clock', 'burst'):
                raise ValueError('Invalid publisher frame pacing')
            self.report['media_frame_pacing'] = pacing
            self.ready.set()
            return
        if not self.ready.is_set():
            raise ValueError('Publisher sent state before startup')
        if kind == 'publisher_done':
            if not self.input_done:
                raise ValueError('Publisher completed before model input EOF')
            self.terminal = True
            return
        if kind != 'webrtc_frames' or not isinstance(record.get('frames'), list) or len(record['frames']) != 1:
            raise ValueError('Unexpected publisher frame record')
        state = record['frames'][0]
        if not isinstance(state, dict):
            raise ValueError('Invalid encoded WebRTC state')
        game = state.get('game')
        validate_game(game, self.last_encoded_frame)
        if game['frame'] >= self.next_frame:
            raise ValueError('Publisher state is ahead of submitted RGB')
        timestamp = state.get('rtp_timestamp')
        if type(timestamp) is not int or not 0 <= timestamp <= 0xFFFFFFFF:
            raise ValueError('Publisher did not report an actual RTP timestamp')
        for key in ('capture_wall_ms', 'encoded_wall_ms'):
            value = state.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'Invalid publisher {key}')
        self.last_encoded_frame = game['frame']
        packet = json_bytes(record) + b'\n'
        with self.lock:
            if len(self.metadata) == self.metadata.maxlen:
                self.report['metadata_evicted'] += 1
            self.metadata.append(packet)
            self.report['media_encoded_frames'] += 1
            self.report['media_encoder_bytes'] = self.report.get('media_encoder_bytes', 0) + size
            self.report.setdefault('first_encoded_byte_ms',
                (time.perf_counter() - self.report['_started_monotonic']) * 1000)

    def _watch(self):
        # Natural EOF is bounded even if the child stalls in native Gst finish.
        while not self.finished.wait(.05) and not self.closing.is_set():
            if self.cancel.is_set():
                self._terminate()
                return
            if self.input_done:
                if not self.finished.wait(self.finish_timeout) and not self.closing.is_set():
                    self._fail(TimeoutError('WebRTC publisher did not finish after RGB EOF'))
                return

    async def stream(self):
        while True:
            self.check()
            with self.lock:
                value = self.metadata.popleft() if self.metadata else None
            if value is not None:
                self.report['media_output_bytes'] = self.report.get('media_output_bytes', 0) + len(value)
                yield value
            elif self.finished.is_set() or self.cancel.is_set():
                self.check()
                return
            else:
                await asyncio.sleep(.01)

    def _close(self):
        with self.close_lock:
            if self.closed:
                return
            self.closing.set()
            self.cancel.set()
            with self.lock:
                self.lock.notify_all()
            if self.process is not None:
                self._terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
                for thread in self.threads:
                    thread.join(timeout=2)
                if any(thread.is_alive() for thread in self.threads):
                    raise RuntimeError('WebRTC publisher pipe workers did not stop')
                for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                    stream.close()
            with self.lock:
                self.frames.clear()
                self.metadata.clear()
                self.report['media_queue_depth'] = 0
            self.closed = True
            self.report['webrtc_cleanup_confirmed'] = True

    async def close(self):
        await asyncio.to_thread(self._close)
