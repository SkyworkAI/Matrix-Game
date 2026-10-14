"""Racing IPC adapter; a bounded input chunk, independent of display consumption."""
import math
import os
from pathlib import Path
import queue
import socket
import subprocess
import threading

import numpy as np

from .game import CastleGame, HEADER


def game_options(value):
    if not isinstance(value, dict) or set(value) - {"mode", "route", "front_drive", "front_grip", "rear_grip"}:
        raise ValueError("Invalid racing options")
    result = dict(mode="time_trial", route="downtown", front_drive=.7, front_grip=1.15, rear_grip=.9)
    result.update(value)
    if result["mode"] not in ("drive", "time_trial", "pursuit") or result["route"] not in ("downtown", "avenue", "skyway"):
        raise ValueError("Unknown racing mode or route")
    for key, lo, hi in (("front_drive", 0, 1), ("front_grip", .9, 1.2), ("rear_grip", .9, 1.2)):
        v = result[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError("Invalid vehicle handling")
    if result["mode"] == "pursuit":
        result["route"] = "avenue"
    return result


def controls(message):
    if not isinstance(message, dict) or set(message) - {"sequence", "drive", "action"}:
        raise ValueError("Invalid racing controls")
    seq = message.get("sequence")
    drive = message.get("drive", [0, 0, 0, 0])
    if type(seq) is not int or not 0 <= seq < 2**53 or not isinstance(drive, list) or len(drive) != 4:
        raise ValueError("Invalid control sequence or drive axes")
    for i, v in enumerate(drive):
        if type(v) not in (int, float) or not math.isfinite(v) or not (-1 if i < 2 else 0) <= v <= 1:
            raise ValueError("Invalid drive axis")
    action = message.get("action")
    if action is not None and action not in ("pause", "resume", "retry", "recover"):
        raise ValueError("Invalid race action")
    return dict(type="controls", sequence=seq, drive=drive, keyboard=True, action=action)


class RacingGame(CastleGame):
    def __init__(self, root, *, fps, log_path, asset_root, options=None, prepare=None):
        self.root, self.asset_root, self.fps = Path(root), Path(asset_root), fps
        self.closed, self.lock = threading.Event(), threading.Lock()
        self.credit, self.queue = threading.Semaphore(1), queue.Queue(maxsize=1)
        self.error = self.ended = None
        self.thread = self.process = self.socket = self.log = None
        self.received, self.prepare = 0, prepare
        self.options = game_options(options or {})
        parent, child = socket.socketpair()
        self.socket = parent
        env = dict(os.environ, CITYSAMPLE_EGL="1") if os.uname().sysname == "Linux" else os.environ.copy()
        command = [str(self.root / "racing/bin/CitySampleReferenceServer"), "--data-dir", str(self.asset_root),
                   "--ipc-fd", str(child.fileno()), "--fps", str(fps), "--block-latents", "1", "--frames", "32765"]
        for key, value in self.options.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        try:
            self.log = open(log_path, "wb")
            self.process = subprocess.Popen(command, env=env, pass_fds=(child.fileno(),), stdout=self.log, stderr=self.log)
            child.close()
            self.socket.settimeout(30)
            hello, payload = self._read()
            expected = dict(type="hello", schema_version=1, width=864, height=480, fps=fps, first_chunk=1, chunk=4,
                            camera_profile="dataset_chase_raised_v2", ground_profile="road_contrast_junction_v1",
                            render_width=1920, render_height=1080)
            if payload or any(hello.get(k) != v for k, v in expected.items()):
                raise ValueError("Racing capture contract mismatch; rebuild the game")
            self.hello = hello
            self.socket.settimeout(None)
            self.send(dict(type="start"))
            self.thread = threading.Thread(target=self._produce, name="racing-reference", daemon=True)
            self.thread.start()
        except BaseException:
            child.close()
            self.close()
            raise

    def _read(self):
        import json
        length, size = HEADER.unpack(self._exact(8))
        if not 0 < length <= 1024 * 1024 or size > 4 * 864 * 480 * 4:
            raise ValueError("Invalid racing IPC packet size")
        return json.loads(self._exact(length)), self._exact(size)

    def send(self, value):
        action = value.get("action")
        if action and self.options["mode"] != "drive":
            super().send(dict(type="race_action", action=action))
        super().send({k: v for k, v in value.items() if k != "action"})

    def _produce(self):
        try:
            while not self.closed.is_set():
                self.credit.acquire()
                if self.closed.is_set():
                    return
                count = 1 if self.received == 0 else 4
                if self.received:
                    self.send(dict(type="ack", frame_end=self.received))
                self.send(dict(type="request", frame_end=self.received + count))
                meta, payload = self._read()
                states = meta.get("telemetry", [])
                caption = meta.get("caption", {})
                if (meta.get("type") != "chunk" or meta.get("frame_start") != self.received + 1
                        or meta.get("frame_end") != self.received + count or meta.get("count") != count
                        or len(payload) != count * 480 * 864 * 4
                        or [s.get("frame") for s in states] != list(range(self.received, self.received + count))
                        or caption.get("video_frame_start") != self.received
                        or caption.get("video_frame_end") != self.received + count - 1
                        or not isinstance(caption.get("prompt"), str) or not 0 < len(caption["prompt"]) <= 8192):
                    raise ValueError("Racing RGB/caption/state alignment mismatch")
                for state in states:
                    if (not math.isfinite(state["simulation_time"])
                            or abs(state["simulation_time"] - (state["frame"] + 1) / self.fps) > .001):
                        raise ValueError("Racing simulation cadence mismatch")
                states[-1]["chunk_prompt"] = caption["prompt"]
                rgba = np.frombuffer(payload, np.uint8).reshape(count, 480, 864, 4)
                if self.prepare is not None:
                    self.prepare.begin()
                    reference = self.prepare.finish([self.prepare.frame(f) for f in rgba])
                else:
                    reference = rgba
                self.received += count
                self.queue.put_nowait((reference, states))
        except BaseException as exc:
            if not self.closed.is_set():
                self.error = exc
