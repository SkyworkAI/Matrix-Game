"""Private native-game IPC with one chunk of lookahead and no input-frame drops."""
import json
import os
from pathlib import Path
import queue
import socket
import struct
import subprocess
import threading

import numpy as np

HEADER = struct.Struct("!II")


class CastleGame:
    def __init__(self, root, *, fps, log_path, seed=1729, scene="castle", expected_prompt=None, asset_root=None,
                 prepare=None, frame_packets=False, capture_mode="color", continuous_pacing=True):
        self.root = Path(root)
        self.asset_root = Path(asset_root) if asset_root is not None else self.root
        self.fps = fps
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.credit = threading.Semaphore(1)
        self.queue = queue.Queue(maxsize=1)
        self.error = None
        self.ended = None
        self.thread = self.process = self.socket = self.log = None
        self.received = 0
        self.prepare = prepare
        self.frame_packets = frame_packets or prepare is not None
        parent, child = socket.socketpair()
        self.socket = parent
        env = os.environ.copy()
        if os.uname().sysname == "Linux":
            env["CASTLE_EGL"] = "1"
        try:
            self.log = open(log_path, "wb")
            self.process = subprocess.Popen(
                [str(self.root / "bin/CastleReferenceServer"), "--data-dir", str(self.asset_root),
                 "--ipc-fd", str(child.fileno()), "--fps", str(fps), "--seed", str(seed), "--scene", scene,
                 "--frame-packets", str(int(self.frame_packets)), "--capture-mode", capture_mode,
                 "--continuous-pacing", str(int(continuous_pacing))],
                env=env, pass_fds=(child.fileno(),), stdout=self.log, stderr=self.log,
            )
            child.close()
            self.socket.settimeout(20)
            hello, payload = self._read()
            expected = dict(type="hello", schema=1, scene=scene, width=1280, height=720, fps=fps, first_chunk=1, chunk=4)
            expected.update(frame_packets=self.frame_packets, capture_mode=capture_mode,
                            continuous_pacing=continuous_pacing)
            if expected_prompt is not None:
                expected.update(prompt=expected_prompt, prompt_verified=True)
            if payload or any(hello.get(key) != value for key, value in expected.items()):
                raise ValueError("Game scene/conditioning handshake mismatch")
            self.hello = hello
            self.socket.settimeout(None)
            self.thread = threading.Thread(target=self._produce, name="castle-reference", daemon=True)
            self.thread.start()
        except BaseException:
            child.close()
            self.close()
            raise

    def _exact(self, size):
        data = bytearray(size)
        view = memoryview(data)
        while view:
            count = self.socket.recv_into(view)
            if not count:
                raise EOFError("Castle game disconnected; inspect its log")
            view = view[count:]
        return data

    def _read(self):
        length, size = HEADER.unpack(self._exact(8))
        if not 0 < length <= 65536 or size > 4 * 1280 * 720 * 4:
            raise ValueError("Invalid castle IPC packet size")
        return json.loads(self._exact(length)), self._exact(size)

    def send(self, value):
        data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        if len(data) > 16384:
            raise ValueError("Control packet too large")
        with self.lock:
            if self.closed.is_set() or self.ended is not None:
                return
            try:
                self.socket.sendall(HEADER.pack(len(data), 0) + data)
            except (BrokenPipeError, ConnectionResetError):
                # Controls can race a terminal packet still being read. The
                # reader owns both normal-end and unexpected-EOF diagnosis.
                if value.get("type") != "controls":
                    raise

    def _produce(self):
        try:
            while not self.closed.is_set():
                self.credit.acquire()
                if self.closed.is_set():
                    return
                self.send(dict(type="request", frame_start=self.received))
                value = self._read_chunk()
                if value is None:
                    return
                self.received += len(value[1])
                self.queue.put_nowait(value)
        except BaseException as exc:
            if not self.closed.is_set():
                self.error = exc

    def _read_chunk(self):
        count = 1 if self.received == 0 else 4
        frames, states = [], []
        if self.prepare is not None:
            self.prepare.begin()
        for offset in (range(count) if self.frame_packets else (0,)):
            meta, payload = self._read()
            if meta.get("type") == "end":
                last = meta.get("frame_start")
                if (payload or meta.get("reason") != "defeated" or type(last) is not int
                        or not self.received <= last < self.received + count
                        or (self.frame_packets and last != self.received + offset)):
                    raise ValueError("Invalid terminal game packet")
                self.ended = meta
                return None
            size = 1 if self.frame_packets else count
            current = meta.get("states", [])
            start = self.received + offset
            if (meta.get("type") != ("frame" if self.frame_packets else "chunk")
                    or meta.get("frame_start") != start or meta.get("count") != size
                    or len(payload) != size * 1280 * 720 * 4
                    or [s.get("frame") for s in current] != list(range(start, start + size))):
                raise ValueError("Castle RGB/state frame alignment mismatch")
            for state in current:
                if not np.isfinite(state["simulation_time"]) or abs(state["simulation_time"] - (state["frame"] + 1) / self.fps) > .001:
                    raise ValueError("Castle reference simulation cadence changed")
            rgba = np.frombuffer(payload, np.uint8).reshape(size, 720, 1280, 4)
            frames.extend(self.prepare.frame(frame) if self.prepare is not None else frame for frame in rgba)
            states.extend(current)
        # Preparation uses the existing one-chunk credit. It cannot prefetch a
        # second block while the consumer is still using the previous block.
        prepared = self.prepare.finish(frames) if self.prepare is not None else (np.stack(frames) if self.frame_packets else rgba)
        return prepared, states

    def next(self, cancel):
        while not cancel.is_set() and not self.closed.is_set():
            if self.error:
                raise RuntimeError(f"Castle reference failed: {self.error}") from self.error
            if self.ended is not None:
                return None
            try:
                value = self.queue.get(timeout=.1)
                self.credit.release()
                return value
            except queue.Empty:
                pass
        return None

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.credit.release()
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.socket.close()
        if self.thread is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError("Castle input reader did not stop")
        if self.process is not None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        if self.log is not None:
            self.log.close()
