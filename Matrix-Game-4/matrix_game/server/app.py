#!/usr/bin/env python3
"""Loopback-only castle game service; native fixed-sink model + WebRTC display."""
import argparse
import asyncio
from collections import deque
import html
import json
import logging
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlsplit

from aiohttp import web
from matrix_game.profiling import ChunkTimer, summarize
from matrix_game.scenes import SCENES, add_model_arguments, resolve_model_arguments
from matrix_game.inference.loader import ModelComponents

LOG = logging.getLogger("castle")


def validate_renderer(hello, allow_software=False):
    name = hello.get("renderer", "unknown")
    if any(kind in name.lower() for kind in ("llvmpipe", "softpipe", "software rasterizer", "swrast")):
        if not allow_software:
            raise RuntimeError("Game is using software OpenGL. Configure the container's GPU graphics "
                               "libraries, or use --allow-software-rendering for correctness tests only.")
        LOG.warning("Software OpenGL enabled for testing: %s", name)
    return name


def display_frames(images, rendered, monitor):
    if not monitor:
        return rendered
    import numpy as np
    reference = np.stack([np.asarray(image) for image in images])
    if reference.shape != rendered.shape:
        raise ValueError("Reference and render dimensions differ")
    return np.concatenate((reference, rendered), axis=2)


class Runtime(ModelComponents):
    def __init__(self, args):
        super().__init__(args)
        self.active = None
        self.lock = asyncio.Lock()
        self.gen = self.vae = self.prompt = None
        self.sink_latent = None
        self.sink_sha256 = None
        self.attention_backend = None



    async def start(self, mode, monitor=False, game_options=None):
        async with self.lock:
            if self.active is not None:
                raise web.HTTPConflict(text="A game session is already active; stop it first.")
            session = Session(self, mode, monitor, game_options)
            self.active = session
            try:
                await session.start()
            except BaseException:
                await session.close()
                self.active = None
                raise
            return session

    async def stop(self, session):
        async with self.lock:
            if self.active is session:
                await session.close()
                self.active = None


class Session:
    def __init__(self, runtime, mode, monitor=False, game_options=None):
        self.runtime, self.mode, self.monitor = runtime, mode, monitor
        self.game_options = game_options
        self.id = uuid.uuid4().hex
        self.cancel = threading.Event()
        self.report = {"_started_monotonic": time.perf_counter(), "frames": 0, "chunk": -1, "mode": mode,
                       "monitor": monitor, "display_width": 1728 if monitor else 864}
        self.report.update(runtime.config())
        self.game = self.output = self.future = self.pump = self.watchdog = None
        self.records = deque(maxlen=120)
        self.profile_rows = deque(maxlen=2048)
        self.serial = 0
        self.socket = None
        self.last_client = time.monotonic()
        self.error = None
        self.closed = False

    async def start(self):
        from matrix_game.server.game import CastleGame
        from matrix_game.server.transport.output import WebRTCModelOutput
        args = self.runtime.args
        prepare = None
        if getattr(args, "reference_pipeline", "stream") == "stream":
            from matrix_game.server.reference_input import ReferencePreprocessor
            prepare = ReferencePreprocessor(model=self.mode == "model", profile=getattr(args, "profile_chunks", False))
        if args.scene == "racing":
            from matrix_game.server.racing import RacingGame
            self.game = await asyncio.to_thread(RacingGame, args.game_root, fps=args.fps,
                asset_root=args.asset_root, options=self.game_options, prepare=prepare,
                log_path=args.runtime / f"game-{self.id}.log")
            self.report.update(game_options=self.game.hello["game_options"], audio_mode="frame_telemetry",
                               capture_profile=self.game.hello)
        else:
            self.game = await asyncio.to_thread(CastleGame, args.game_root, fps=args.fps,
                                           asset_root=args.asset_root,
                                           scene=args.scene, expected_prompt=args.prompt,
                                           prepare=prepare,
                                           log_path=args.runtime / f"game-{self.id}.log")
            self.report["audio_mode"] = "frame_events"
        if args.scene != "racing" and self.game.hello.get("audio_schema") != 1:
            raise RuntimeError("Rebuild CastleReferenceServer: frame-synchronized audio schema 1 is required")
        if args.scene != "racing" and self.game.hello.get("mouse_input_schema") != 2:
            raise RuntimeError("Rebuild CastleReferenceServer: physics-tick mouse input schema 2 is required")
        self.report["game_renderer"] = validate_renderer(
            self.game.hello, getattr(args, "allow_software_rendering", False))
        self.report["trees"] = self.game.hello.get("trees")
        self.report.update(audio_schema=self.game.hello.get("audio_schema"),
                           mouse_input_schema=self.game.hello.get("mouse_input_schema"),
                           camera_max_degrees_per_second=self.game.hello.get("camera_max_degrees_per_second"),
                           fade_small_edges=self.game.hello.get("fade_small_edges"),
                           road_surface_splits=self.game.hello.get("road_surface_splits"))
        self.output = WebRTCModelOutput(self.report["display_width"], 480, args.fps, self.id, self.cancel, self.report, queue_size=4)
        await self.output.start()
        self.future = asyncio.get_running_loop().run_in_executor(None, self.produce)
        self.future.add_done_callback(self.output.inference_done)
        self.pump = asyncio.create_task(self.metadata())
        # RTSP is not advertised until a first encoded frame has reached it.
        deadline = time.monotonic() + 45
        while not self.report["media_encoded_frames"]:
            self.output.check()
            if self.future.done():
                await self.future
                raise RuntimeError("Inference ended before the first frame")
            if time.monotonic() > deadline:
                raise TimeoutError("No encoded first frame after 45 seconds")
            await asyncio.sleep(.02)
        self.last_client = time.monotonic()
        self.watchdog = asyncio.create_task(self.watch())
        LOG.info("Session %s started (%s, %s)", self.id, self.mode,
                 {k: v for k, v in self.game.hello.items() if k != "pursuit_map"})

    def produce(self):
        import numpy as np
        from PIL import Image
        from matrix_game.inference.preprocessing import crop_and_resize, frames_to_tensor, tensor_to_uint8
        from matrix_game.server.reference_input import PreparedReference
        stream = None
        profiling = getattr(self.runtime.args, "profile_chunks", False)
        try:
            if self.mode == "model":
                stream = self.runtime.stream(profile=profiling)
            if profiling and self.mode == "model":
                import torch
                self.report["profile_torch_threads"] = torch.get_num_threads()
            chunks, generated, inference_time = 0, 0, 0.
            current_prompt = self.runtime.args.prompt
            prompt_updates = 0
            encoder = self.runtime.caption_encoder
            initial_encodes, initial_hits = (encoder.encodes, encoder.hits) if encoder else (0, 0)
            wall_start = wall_frames = None
            while not self.cancel.is_set():
                timer = ChunkTimer(profiling)
                cycle_started = time.perf_counter()
                with timer.measure("wait_reference_ms"):
                    value = self.game.next(self.cancel)
                if value is None:
                    break
                reference, states = value
                started = time.perf_counter()
                prepared = isinstance(reference, PreparedReference)
                if prepared:
                    images, pixels = reference.images, reference.pixels
                    if profiling:
                        timer.values.update({"background_" + key: val for key, val in reference.timings_ms.items()})
                else:
                    # Retained serial path for pixel-exact A/B and rollback.
                    images = []
                    for frame in reference:
                        with timer.measure("rgba_to_rgb_ms"):
                            image = Image.fromarray(frame).convert("RGB")
                        with timer.measure("resize_crop_ms"):
                            images.append(crop_and_resize(image, 480, 864))
                def publish(rgb):
                    if len(rgb) != len(states):
                        raise ValueError("Decoded frame count differs from reference metadata")
                    if not self.cancel.is_set():
                        with timer.measure("publish_compose_ms"):
                            display = display_frames(images, rgb, self.monitor)
                        with timer.measure("publish_metadata_ms"):
                            self.output.note_frames(len(display), chunk_index=chunks, game_frames=states)
                        with timer.measure("publish_pack_ms"):
                            packed = display.tobytes()
                        with timer.measure("publish_enqueue_ms"):
                            self.output(packed)
                        self.report["rgb_ready_ms"] = (time.perf_counter() - started) * 1000

                if stream is not None:
                    if self.runtime.args.prompt_mode == "dynamic" and chunks % self.runtime.args.prompt_interval_chunks == 0:
                        candidate = states[-1].get("chunk_prompt")
                        if not candidate:
                            raise ValueError("Dynamic mode requires a frame-aligned native caption")
                        if candidate != current_prompt:
                            with timer.measure("prompt_submit_ms"):
                                embedded = self.runtime.caption_encoder(candidate)
                                stream.set_prompt(embedded)
                            current_prompt = candidate
                            prompt_updates += 1
                    self.report.update(prompt=current_prompt, prompt_updates=prompt_updates,
                        prompt_encodes=encoder.encodes - initial_encodes if encoder else 0,
                        prompt_cache_hits=encoder.hits - initial_hits if encoder else 0)
                    early = getattr(self.runtime.args, "early_publish", False)
                    if not prepared:
                        with timer.measure("prepare_tensor_ms"):
                            pixels = frames_to_tensor(images)
                    with timer.measure("stream_wall_ms"):
                        chunk = stream.push(pixels, frame_start=states[0]["frame"],
                                            on_rgb=(lambda value: publish(value.numpy())) if early else None)
                    with timer.measure("materialize_rgb_ms"):
                        rgb = chunk.rgb.numpy() if early else tensor_to_uint8(chunk.pixels)
                    if profiling:
                        timer.values.update({"cuda_" + key + "_ms": value for key, value in chunk.timings_ms.items()})
                    if not early:
                        publish(rgb)
                else:
                    rgb = np.stack([np.asarray(image) for image in images])
                    publish(rgb)
                elapsed = time.perf_counter() - started
                if self.cancel.is_set():
                    break
                if chunks >= 2:
                    generated += len(rgb)
                    inference_time += elapsed
                chunks += 1
                if chunks == 25:
                    wall_start, wall_frames = time.perf_counter(), states[-1]["frame"] + 1
                if wall_start is not None:
                    seconds = time.perf_counter() - wall_start
                    self.report["online_fps"] = (states[-1]["frame"] + 1 - wall_frames) / seconds if seconds else 0
                if profiling:
                    row = dict(timer.values, chunk=chunks - 1, frame_count=len(rgb),
                               processing_ms=elapsed * 1000,
                               cycle_ms=(time.perf_counter() - cycle_started) * 1000)
                    if all("capture_ms" in state for state in states):
                        row["native_capture_ms"] = sum(state["capture_ms"] for state in states)
                    self.profile_rows.append(row)
                    self.report["profile_last"] = row
                self.report.update(frames=states[-1]["frame"] + 1, chunk=chunks - 1,
                                   model_fps=generated / inference_time if inference_time else 0,
                                   chunk_ms=elapsed * 1000, simulation_time=states[-1]["simulation_time"])
        except BaseException:
            if not self.cancel.is_set():
                raise
        finally:
            if stream is not None:
                stream.close()

    async def metadata(self):
        try:
            async for raw in self.output.stream():
                self.serial += 1
                record = json.loads(raw)
                record.update(generation_id=self.id, serial=self.serial)
                self.records.append(record)
        except Exception as exc:
            self.error = str(exc)
            LOG.exception("Session media failed")

    async def watch(self):
        try:
            while not self.closed:
                if self.future.done():
                    exc = self.future.exception()
                    ended = getattr(self.game, "ended", None)
                    self.error = str(exc) if exc else (None if ended else "Session finished")
                    if not exc and ended:
                        self.report["ended"] = ended["reason"]
                        if self.socket is not None and not self.socket.closed:
                            await asyncio.wait_for(self.socket.send_json(dict(type="ended", reason=ended["reason"])), 2)
                    if exc:
                        LOG.error("Session inference failed", exc_info=(type(exc), exc, exc.__traceback__))
                    break
                if time.monotonic() - self.last_client > 15:
                    self.error = "Browser disconnected"
                    break
                await asyncio.sleep(.25)
        finally:
            if not self.closed:
                asyncio.create_task(self.runtime.stop(self))

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel.set()
        if self.game is not None:
            await asyncio.to_thread(self.game.close)
        if self.future is not None:
            await asyncio.gather(asyncio.shield(self.future), return_exceptions=True)
        if self.output is not None:
            await self.output.close()
        for task in (self.watchdog, self.pump):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if self.socket is not None and not self.socket.closed:
            await self.socket.close(message=b"Session stopped")
        path = self.runtime.args.runtime / f"session-{self.id}.json"
        path.write_text(json.dumps(dict(self.report, error=self.error), indent=2))
        if self.profile_rows:
            profile_path = self.runtime.args.runtime / f"profile-{self.id}.json"
            profile_path.write_text(json.dumps(dict(config=self.runtime.config(),
                summary=summarize(self.profile_rows), rows=list(self.profile_rows)), indent=2))
        LOG.info("Session %s stopped: %s", self.id, self.report)


def controls(message):
    allowed = {"sequence", "forward", "right", "look_x", "look_y", "look_reset", "fire", "ads", "shield", "crouch", "weapon", "actions"}
    if not isinstance(message, dict) or set(message) - allowed:
        raise ValueError("Invalid controls")
    import math
    for key, limit in (("forward", 1), ("right", 1), ("look_x", 2000), ("look_y", 2000)):
        val = message.get(key, 0)
        if type(val) not in (float, int) or not math.isfinite(val) or abs(val) > limit:
            raise ValueError("Invalid axis")
    for key in ("fire", "ads", "shield", "crouch", "look_reset"):
        if type(message.get(key, False)) is not bool:
            raise ValueError("Invalid button")
    if type(message.get("sequence")) is not int or not 0 <= message["sequence"] < 2**53:
        raise ValueError("Invalid sequence")
    if type(message.get("weapon", -1)) is not int or not -1 <= message.get("weapon", -1) <= 2:
        raise ValueError("Invalid weapon")
    actions = message.get("actions", [])
    if not isinstance(actions, list) or len(actions) > 8 or any(a not in ("jump", "dash", "reload", "trap", "cover") for a in actions):
        raise ValueError("Invalid action")
    return dict(message, type="controls")


def application(runtime):
    @web.middleware
    async def origin_guard(request, handler):
        host = urlsplit("http://" + request.host).hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise web.HTTPForbidden(text="Only loopback hosts are supported")
        origin = request.headers.get("Origin")
        if origin and origin != f"http://{request.host}":
            raise web.HTTPForbidden(text="Different-origin request rejected")
        if request.method == "POST" and request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType(text="Expected application/json")
        return await handler(request)

    app = web.Application(middlewares=[origin_guard], client_max_size=16384)
    async def index(request):
        template = (Path(__file__).parent.parent / "web/index.html").read_text()
        name = SCENES[getattr(runtime.args, "scene", "castle")]["name"]
        return web.Response(text=template.replace("__SCENE_NAME__", html.escape(name)).replace(
                            "__GAME_KIND__", "racing" if getattr(runtime.args, "scene", "castle") == "racing" else "shooter"),
                            content_type="text/html", headers={"Cache-Control": "no-store"})
    async def reader(request):
        return web.FileResponse(Path(__file__).parent.parent / "web/mediamtx-reader.js")
    async def hud_script(request):
        return web.FileResponse(Path(__file__).parent.parent / "web/hud.js", headers={"Cache-Control": "no-store"})
    async def audio_script(request):
        return web.FileResponse(Path(__file__).parent.parent / "web/audio.js", headers={"Cache-Control": "no-store"})
    async def audio_sample(request):
        return web.FileResponse(runtime.args.asset_root / "assets/combat-audio" / (request.match_info["sound"] + ".wav"),
                                headers={"Cache-Control": "no-cache"})
    async def racing_script(request):
        return web.FileResponse(Path(__file__).parent.parent / "web/racing.js", headers={"Cache-Control": "no-store"})
    async def racing_audio(request):
        return web.FileResponse(runtime.args.asset_root / "assets/audio" / (request.match_info["sound"] + ".wav"))
    async def license_file(request):
        return web.FileResponse(Path(__file__).parent.parent / "web/mediamtx-reader.LICENSE")
    async def status(request):
        session = runtime.active
        return web.json_response(dict(ready=True, fps=runtime.args.fps, active=session.id if session else None,
                                      config=runtime.config(),
                                      report=dict(session.report) if session else None))
    async def start(request):
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise web.HTTPBadRequest(text="Invalid JSON") from None
        if not isinstance(body, dict) or set(body) - {"mode", "monitor", "game_options"}:
            raise web.HTTPBadRequest(text="Invalid session options")
        mode = body.get("mode", "model")
        if mode not in ("model", "reference") or (mode == "model" and runtime.gen is None):
            raise web.HTTPBadRequest(text="Model not available")
        monitor = body.get("monitor", False)
        if type(monitor) is not bool or (monitor and mode != "model"):
            raise web.HTTPBadRequest(text="Dual view requires model mode")
        options = body.get("game_options")
        if getattr(runtime.args, "scene", "castle") == "racing":
            from matrix_game.server.racing import game_options
            try:
                options = game_options(options or {})
            except ValueError as exc:
                raise web.HTTPBadRequest(text=str(exc)) from None
            session = await runtime.start(mode, monitor, options)
        else:
            if options is not None:
                raise web.HTTPBadRequest(text="Racing options are not valid for this scene")
            session = await runtime.start(mode, monitor)
        return web.json_response(dict(id=session.id, fps=runtime.args.fps, monitor=session.monitor,
                                     whep=f"http://127.0.0.1:{runtime.args.whep_port}/native-{session.id}/whep"))
    def get_session(request):
        session = runtime.active
        if session is None or session.id != request.match_info["id"]:
            raise web.HTTPGone(text="Session is no longer active")
        return session
    async def stop(request):
        await runtime.stop(get_session(request))
        return web.json_response({"stopped": True})
    async def socket(request):
        session = get_session(request)
        if session.socket is not None:
            raise web.HTTPConflict(text="This session already has a controller")
        ws = web.WebSocketResponse(max_msg_size=16384, heartbeat=5)
        await ws.prepare(request)
        session.socket = ws
        session.last_client = time.monotonic()
        async def send():
            serial, last_stats = 0, 0.
            while not session.closed:
                for record in list(session.records):
                    if record["serial"] > serial:
                        await asyncio.wait_for(ws.send_json(record), 2)
                        serial = record["serial"]
                if time.monotonic() - last_stats >= .5:
                    await asyncio.wait_for(ws.send_json(dict(type="stats", report=dict(session.report))), 2)
                    last_stats = time.monotonic()
                await asyncio.sleep(.02)
        sender = asyncio.create_task(send())
        sender.add_done_callback(lambda task: asyncio.create_task(ws.close()) if not task.cancelled() and task.exception() else None)
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    validate = controls
                    if runtime.args.scene == "racing":
                        from matrix_game.server.racing import controls as validate
                    value = validate(json.loads(msg.data))
                    session.last_client = time.monotonic()
                    await asyncio.to_thread(session.game.send, value)
        except (ValueError, json.JSONDecodeError):
            await ws.close(code=1008, message=b"Invalid controls")
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            asyncio.create_task(runtime.stop(session))
        return ws
    async def cleanup(app):
        if runtime.active:
            await runtime.stop(runtime.active)
    app.router.add_get("/", index)
    app.router.add_get("/mediamtx-reader.js", reader)
    app.router.add_get("/hud.js", hud_script)
    app.router.add_get("/audio.js", audio_script)
    app.router.add_get("/racing.js", racing_script)
    app.router.add_get(r"/racing-audio/{sound:engine-loop|tire-loop|tire-front-loop|impact-medium|impact-heavy}.wav", racing_audio)
    app.router.add_get(r"/audio/{sound:[0-6]}.wav", audio_sample)
    app.router.add_get("/mediamtx-reader.LICENSE", license_file)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/start", start)
    app.router.add_post("/api/{id}/stop", stop)
    app.router.add_get("/api/{id}/ws", socket)
    app.on_cleanup.append(cleanup)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--game-root", type=Path, default=Path("game"))
    parser.add_argument("--runtime", type=Path, default=Path("runtime"))
    parser.add_argument("--fps", type=int, default=40)
    parser.add_argument("--port", type=int, default=8025)
    parser.add_argument("--whep-port", type=int, default=8890)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--allow-software-rendering", action="store_true", help="Permit slow software OpenGL for correctness tests")
    parser.add_argument("--early-publish", action=argparse.BooleanOptionalAction, default=True, help="Publish RGB before waiting for clean KV")
    parser.add_argument("--profile-chunks", action="store_true", help="Save bounded per-chunk CPU/CUDA timings on session close")
    parser.add_argument("--reference-pipeline", choices=("serial", "stream"), default="stream",
                        help="Prepare arriving frames in the one-block input worker, or retain serial preprocessing")
    args = parser.parse_args(argv)
    try:
        resolve_model_arguments(args, require_model=not args.reference_only)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if not 1 <= args.fps <= 60 or (args.scene != "racing" and 120 % args.fps):
        parser.error("FPS must be 1..60, and divide 120 for shooters")
    if args.scene == "racing":
        sounds = [args.asset_root / "assets/audio" / (name + ".wav") for name in
                  ("engine-loop", "tire-loop", "tire-front-loop", "impact-medium", "impact-heavy")]
        binary = args.game_root / "racing/bin/CitySampleReferenceServer"
    else:
        sounds = [args.asset_root / "assets/combat-audio" / f"{i}.wav" for i in range(7)]
        binary = args.game_root / "bin/CastleReferenceServer"
    if any(not p.is_file() for p in sounds):
        parser.error("Missing scene sound bank; download game assets")
    if not binary.is_file():
        parser.error("Build the scene's native game first; see docs/installation.md")
    for values in (args.steps, args.first_steps):
        if not values or values[0] != 1000 or any(not 0 < v <= 1000 for v in values) or any(a <= b for a, b in zip(values, values[1:])):
            parser.error("Timesteps must decrease strictly from 1000")
    args.runtime.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = Runtime(args)
    if not args.reference_only:
        runtime.load()
    web.run_app(application(runtime), host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
