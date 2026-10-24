"""CPU contract tests; fake compute is not neural-generation acceptance."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome
from reactor_runtime.distributed import DistributedRunner
from reactor_runtime.interface.model.contract import ContractError, ModelContract

import matrix_game4
import matrix_game4_assets
from matrix_game4 import MatrixGame4
from matrix_game4_backend import MAX_CHUNKS, MatrixGame4Backend
from matrix_game4_game import NativeGame, NativeGameError, ReferenceChunk
from matrix_game4_model import (
    InvalidStep,
    MatrixGame4Input,
    MatrixGame4Model,
    MatrixGame4Result,
    WorldStart,
)
from matrix_game4_types import GameFinished, MatrixGame4State


def rgb(count, value=0):
    return np.full((count, 480, 864, 3), value, np.uint8)


class FakeBackend:
    def __init__(self):
        self.starts = []
        self.calls = []
        self.closes = 0
        self.index = 0
        self.error = None

    def start(self, seed):
        self.starts.append(seed)
        self.index = 0

    def generate(self, frames, frame_start, caption):
        self.calls.append((frames, frame_start, caption))
        if self.error:
            raise self.error
        index = self.index
        self.index += 1
        return 255 - frames, index, frame_start, frame_start + len(frames) - 1

    def close(self):
        self.closes += 1


class FakeGame:
    def __init__(self, *args):
        self.frame = 0
        self.packets = []
        self.closed = False
        self.finish = False

    def send(self, packet):
        self.packets.append(packet)

    def next(self):
        if self.finish:
            return None
        count = 1 if self.frame == 0 else 4
        states = [
            {"frame": i, "chunk_prompt": f"caption-{i // 9}"}
            for i in range(self.frame, self.frame + count)
        ]
        frames = rgb(count, self.frame % 255)
        self.frame += count
        return ReferenceChunk(frames=frames, states=states)

    def close(self):
        self.closed = True


class CPUWorker(MatrixGame4Model):
    """Exercise the actual runner's spawn and CPU-array transport without weights."""

    def load(self):
        self._backend = FakeBackend()
        self._backends = {"ancient-egypt": self._backend}


@pytest.fixture
def app(monkeypatch):
    value = MatrixGame4()
    value.state = MatrixGame4State()
    value._settings = {
        scene: SimpleNamespace(
            scene=scene, prompt_mode="fixed", prompt_interval_chunks=1
        )
        for scene in ("ancient-egypt", "castle", "racing")
    }
    value._engine = MatrixGame4Model()
    value._engine._backend = FakeBackend()
    value._engine._backends = {
        "ancient-egypt": value._engine._backend,
        "castle": FakeBackend(),
        "racing": FakeBackend(),
    }
    value._engine.healthy = True
    value._engine.shutdown = Mock()
    value._game = None
    value.output.flush = Mock()
    value.messages = []

    async def send(message):
        value.messages.append(message)

    value.send = send
    monkeypatch.setattr(matrix_game4, "NativeGame", FakeGame)
    monkeypatch.setattr(
        matrix_game4,
        "control_packet",
        lambda state, scene, sequence, **kw: {
            "type": "controls",
            "sequence": sequence,
            "forward": state._forward,
            **kw,
        },
    )
    return value


async def step(app):
    input = await app.process_input()
    try:
        result = app.generate(input)
    except Exception as error:  # noqa: BLE001 - emulate the Runtime outcome boundary
        outcome = StepOutcome(error=error, elapsed=0.1)
    else:
        outcome = StepOutcome(result=result, elapsed=0.25)
    return input, await app.process_output(outcome)


async def command(app, name, **arguments):
    contract = ModelContract.of(MatrixGame4)
    payload = contract.validate(name, arguments)
    handler = contract.commands[name]
    result = handler.handler(app, **asdict(payload))
    return await result if handler.is_async else result


def test_explicit_start_and_pause_refuse_without_model(app):
    async def run():
        with pytest.raises(ApplicationError):
            await step(app)
        await app.start()
        app.state.paused = True
        with pytest.raises(ApplicationError):
            await step(app)
        assert app._engine._backend.calls == []
        assert app._game.frame == 0

    asyncio.run(run())


def test_application_hooks_against_scripted_model_half(app):
    async def run():
        engine = Mock(spec=["generate", "reset", "shutdown", "healthy"])
        engine.healthy = True
        app._engine = engine
        with pytest.raises(ApplicationError):
            await app.process_input()
        engine.generate.assert_not_called()

        await app.start()
        input = await app.process_input()
        assert input.start == WorldStart(seed=0, scene="ancient-egypt")
        engine.generate.return_value = MatrixGame4Result(
            world_id=input.world_id,
            chunk_index=1,
            frame_start=0,
            frame_end=0,
            frames=rgb(1, 73),
            frames_wanted=4,
            complete=False,
            scene="ancient-egypt",
        )
        result = app.generate(input)
        engine.generate.assert_called_once_with(input)
        output = await app.process_output(StepOutcome(result=result, elapsed=0.375))
        assert np.array_equal(output.main_video[:, :, :864], input.frames)
        assert np.all(output.main_video[:, :, 864:] == 73)
        assert app.state._applied_world_id == input.world_id
        assert app.state._completed_chunks == 1
        assert app.state._last_seconds == 0.375

        await app.set_paused(True)
        with pytest.raises(ApplicationError):
            await app.process_input()
        assert engine.generate.call_count == 1
        await app.set_paused(False)
        next_input = await app.process_input()
        assert next_input.start is None and next_input.frame_start == 1
        assert next_input.frames.shape == (4, 480, 864, 3)
        failure = RuntimeError("scripted model failure")
        engine.generate.side_effect = failure
        with pytest.raises(RuntimeError) as caught:
            app.generate(next_input)
        assert caught.value is failure
        messages_before = len(app.messages)
        with pytest.raises(RuntimeError) as caught:
            await app.process_output(StepOutcome(error=failure, elapsed=0.1))
        assert caught.value is failure
        assert app.state._completed_chunks == 1 and app.state._frame_end == 0
        assert len(app.messages) == messages_before
        engine.shutdown.assert_called_once_with()

    asyncio.run(run())


def test_ten_chunks_are_frame_paired_and_acknowledge_start(app):
    async def run():
        app.state.seed = 29
        await app.start()
        for i in range(10):
            input, output = await step(app)
            assert (input.start is not None) == (i == 0)
            assert output.main_video.shape == (1 if i == 0 else 4, 480, 1728, 3)
            assert np.array_equal(output.main_video[:, :, :864], input.frames)
            assert np.array_equal(output.main_video[:, :, 864:], 255 - input.frames)
            assert [
                row["game"]["frame"] for row in output.__metadata__["main_video"]
            ] == list(range(input.frame_start, input.frame_start + len(input.frames)))
        assert app._engine._backend.starts == [29]
        assert app.state._completed_chunks == 10
        assert app.state._frame_end == 36
        assert app.state._last_seconds == 0.25

    asyncio.run(run())


def test_generate_does_not_read_application_state(app):
    async def run():
        await app.start()
        input = await app.process_input()
        app.state = None
        assert app.generate(input).chunk_index == 1

    asyncio.run(run())


def test_failure_never_advances_completed_count(app):
    async def run():
        await app.start()
        await step(app)
        game = app._game
        app._engine._backend.error = RuntimeError("native compute failed")
        with pytest.raises(RuntimeError, match="native compute failed"):
            await step(app)
        assert game.closed
        assert app.state._completed_chunks == 1
        assert app.state._frame_end == 0
        assert app._engine._failed

    asyncio.run(run())


def test_new_world_and_session_seed_isolation(app):
    async def run():
        await app.start()
        await step(app)
        old_game = app._game
        app.state.seed = 123
        await app.start()
        input, _ = await step(app)
        assert old_game.closed
        assert input.world_id == 2 and input.frame_start == 0
        assert input.start.seed == 123
        assert app._engine._backend.starts == [0, 123]
        await app.on_session_ended()
        app.state = MatrixGame4State()  # Runtime creates a new state per session.
        await app.start()
        input, _ = await step(app)
        assert input.start.seed == 0

    asyncio.run(run())


def test_pause_keeps_native_history_and_caption_transition(app):
    async def run():
        app._settings["racing"].prompt_mode = "dynamic"
        app.state.scene = "racing"
        await app.start()
        await step(app)
        app.state.paused = True
        with pytest.raises(ApplicationError):
            await step(app)
        app.state.paused = False
        for _ in range(9):
            await step(app)
        assert len({call[2] for call in app._engine._backend.calls}) >= 3
        assert app._engine._backend.starts == [0]
        assert app.state._world_id == 1

    asyncio.run(run())


def test_control_refusal_and_awaited_reply(app):
    async def run():
        with pytest.raises(CommandError) as caught:
            await command(app, "set_controls", forward=1)
        assert caught.value.code == "not_running"
        assert app.messages == []
        await app.start()
        reply = await command(app, "set_controls", forward=1, fire=True)
        assert reply.forward == 1 and reply.fire
        assert app.messages[-1] == reply
        app.look(10, -4)
        assert app._game.packets[-1]["look_x"] == 10
        app.action("jump")
        assert app._game.packets[-1]["actions"] == ["jump"]
        with pytest.raises(CommandError) as caught:
            app.action("retry")
        assert caught.value.code == "wrong_scene"

    asyncio.run(run())


def test_another_viewer_leaving_does_not_reset_the_game(app):
    async def run():
        await app.start()
        await step(app)
        app.connected = asyncio.Event()
        app.connected.set()
        app.state._forward = 1
        game = app._game
        await app.on_disconnected(None)
        assert app.state._forward == 1
        assert app._game is game and not game.closed
        app.connected.clear()
        await app.on_disconnected(None)
        assert app.state._forward == 0
        assert app._engine._backend.closes == 0
        assert not game.closed

    asyncio.run(run())


def test_native_end_is_reported_without_inventing_a_frame(app):
    async def run():
        await app.start()
        await step(app)
        app._game.finish = True
        _, output = await step(app)
        assert output is None
        assert not app.state._running
        assert app.state._completed_chunks == 1
        assert isinstance(app.messages[-1], GameFinished)
        assert app.messages[-1].reason == "native_game_ended"

    asyncio.run(run())


@pytest.mark.parametrize(
    "fault", ["frame_id", "shape", "dtype", "caption", "capture", "control"]
)
def test_game_input_fault_ends_world_and_allows_restart(app, fault):
    async def run():
        app.state.scene = "racing"
        app._settings["racing"].prompt_mode = "dynamic"
        await app.start()
        await step(app)
        game = app._game
        backend = app._engine._backend
        reference = game.next()
        if fault == "frame_id":
            reference.states[0]["frame"] = 1000
        elif fault == "shape":
            reference = replace(reference, frames=rgb(1))
        elif fault == "dtype":
            reference = replace(reference, frames=reference.frames.astype(np.float32))
        elif fault == "caption":
            reference.states[-1].pop("chunk_prompt")
        game.next = Mock(return_value=reference)
        if fault == "capture":
            game.next.side_effect = NativeGameError("lost capture")
        elif fault == "control":
            game.send = Mock(side_effect=NativeGameError("lost control connection"))
        with pytest.raises(ApplicationError, match="Game input failed"):
            await step(app)
        assert game.closed and app._game is None
        assert not app.state._running and app.state._reference is None
        assert app.state._completed_chunks == 1
        assert app.state._frame_end == 0
        assert len(backend.calls) == 1 and backend.closes == 1
        assert app.messages[-1] == GameFinished(world_id=1, reason="native_game_error")
        assert app.messages[-2].running is False
        app._engine.shutdown.assert_not_called()
        with pytest.raises(ApplicationError):
            await step(app)
        await app.start()
        for _ in range(10):
            await step(app)
        assert app.state._world_id == 2 and app.state._completed_chunks == 10
        assert app._engine._backend is backend

    asyncio.run(run())


def test_model_contract_and_errors():
    model = MatrixGame4Model()
    model._backend = FakeBackend()
    model._backends = {"ancient-egypt": model._backend}
    input = MatrixGame4Input(
        world_id=1,
        start=WorldStart(seed=0, scene="ancient-egypt"),
        frames=rgb(1),
        frame_start=0,
    )
    with pytest.raises(FrozenInstanceError):
        input.world_id = 2
    with pytest.raises(InvalidStep, match="needs its seed"):
        model.generate(replace(input, start=None))
    with pytest.raises(InvalidStep, match="Expected one"):
        model.generate(replace(input, frames=rgb(4)))
    result = model.generate(input)
    assert result.chunk_index == 1
    assert not hasattr(result, "input")
    with pytest.raises(InvalidStep, match="Nonconsecutive"):
        model.generate(replace(input, start=None, frames=rgb(4)))
    model._backend.error = RuntimeError("fail")
    next_input = replace(input, start=None, frames=rgb(4), frame_start=1)
    with pytest.raises(RuntimeError, match="fail"):
        model.generate(next_input)
    model._backend.error = None
    with pytest.raises(InvalidStep, match="previous chunk failed"):
        model.generate(next_input)
    assert model.generate(replace(input, world_id=2)).chunk_index == 1
    model.reset()
    assert model._world_id is None and model._next_frame == 0


@pytest.mark.parametrize("fail_at", [None, 3])
def test_real_runtime_step_loop(app, fail_at):
    async def run():
        app._on_loop_ready()
        app._sessions_started = 1
        await app.start()
        emitted = []
        reports = []
        done = asyncio.Event()

        async def emit(output, *, compute_time=None):
            emitted.append((output, compute_time))

        def report(value, session, rate):
            reports.append(value)
            if len(reports) == fail_at:
                app._engine._backend.error = RuntimeError("real loop failure")
            if len(reports) == 10:
                app._live.clear()
                done.set()

        app.emit = emit
        app._report_step = report
        app._live.set()
        task = asyncio.create_task(app.run())
        try:
            if fail_at is None:
                await asyncio.wait_for(done.wait(), timeout=5)
                assert len(reports) == len(emitted) == 10
                # Fixed native capture cadence, not model-compute-based playout.
                assert all(value[1] is None for value in emitted)
            else:
                with pytest.raises(RuntimeError, match="real loop failure"):
                    await asyncio.wait_for(task, timeout=5)
                assert len(reports) == len(emitted) == fail_at
                assert app.state._completed_chunks == fail_at
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(run())


def test_model_dependency_graph_imports_without_runtime_or_torch():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('reactor_runtime', 'torch', 'matrix_game'):
            raise ImportError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from matrix_game4_model import MatrixGame4Model
assert MatrixGame4Model()._world_id is None
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_real_distributed_runner_preserves_frame_order(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    runner = DistributedRunner(
        CPUWorker, world_size=1, call_timeout=15, start_timeout=30
    )
    try:
        runner.start()
        frame = 0
        for index in range(10):
            count = 1 if index == 0 else 4
            result = runner.generate(
                MatrixGame4Input(
                    world_id=1,
                    start=WorldStart(seed=17, scene="ancient-egypt")
                    if index == 0
                    else None,
                    frames=rgb(count, index),
                    frame_start=frame,
                )
            )
            assert result.chunk_index == index + 1
            assert result.frame_start == frame and result.frame_end == frame + count - 1
            assert np.all(result.frames == 255 - index)
            frame += count
        runner.reset()
        result = runner.generate(
            MatrixGame4Input(
                world_id=1,
                start=WorldStart(seed=8, scene="ancient-egypt"),
                frames=rgb(1),
                frame_start=0,
            )
        )
        assert result.chunk_index == 1
    finally:
        runner.shutdown()
    assert not runner.healthy


def test_pause_reply_broadcasts_and_releases_held_buttons(app):
    async def run():
        await app.start()
        await command(app, "set_controls", forward=1, fire=True)
        reply = await app.set_paused(True)
        assert reply.paused is True
        assert reply.forward == 0
        assert reply.fire is False
        assert app.messages[-1] == reply
        await app.set_paused(False)
        assert not app.state.paused

    asyncio.run(run())


def test_native_horizon_finishes_without_silent_restart():
    model = MatrixGame4Model()
    model._backend = FakeBackend()
    model._backends = {"ancient-egypt": model._backend}
    model._scene = "ancient-egypt"
    model._world_id = 5
    model._chunk_index = model._backend.index = 8191
    model._next_frame = 1 + 4 * 8190
    result = model.generate(
        MatrixGame4Input(
            world_id=5, start=None, frames=rgb(4), frame_start=model._next_frame
        )
    )
    assert result.complete and result.frames_wanted == 0
    assert result.chunk_index == 8192 and result.frame_end == 32764
    assert model._backend.starts == []
    with pytest.raises(InvalidStep, match="exhausted"):
        model.generate(
            MatrixGame4Input(world_id=5, start=None, frames=rgb(4), frame_start=32765)
        )


def test_race_action_refusal_uses_active_mode_not_next_start(app):
    async def run():
        app.state.scene = "racing"
        app.state.race_mode = "drive"
        await app.start()
        app.state.race_mode = "time_trial"
        with pytest.raises(CommandError, match="time trial or pursuit"):
            app.action("recover")
        await app.start()
        app.state.race_mode = "drive"
        app.action("recover")
        assert app._game.packets[-1]["action"] == "recover"
        app.state.race_mode = "pursuit"
        await app.start()
        app.state.race_mode = "time_trial"
        before = len(app._game.packets)
        with pytest.raises(CommandError, match="disabled in native pursuit"):
            app.action("recover")
        assert len(app._game.packets) == before
        app.action("pause")
        assert app._game.packets[-1]["action"] == "pause"

    asyncio.run(run())


def test_start_failure_does_not_allow_using_the_old_stream():
    model = MatrixGame4Model()
    model._backend = FakeBackend()
    model._backends = {"ancient-egypt": model._backend}
    model.generate(
        MatrixGame4Input(
            world_id=1,
            start=WorldStart(seed=1, scene="ancient-egypt"),
            frames=rgb(1),
            frame_start=0,
        )
    )
    model._backend.start = Mock(side_effect=RuntimeError("start failed"))
    with pytest.raises(RuntimeError, match="start failed"):
        model.generate(
            MatrixGame4Input(
                world_id=2,
                start=WorldStart(seed=2, scene="ancient-egypt"),
                frames=rgb(1),
                frame_start=0,
            )
        )
    with pytest.raises(InvalidStep, match="previous chunk failed"):
        model.generate(
            MatrixGame4Input(world_id=1, start=None, frames=rgb(4), frame_start=1)
        )
    assert model._chunk_index == 1


def test_playout_matches_native_game_clock():
    assert MatrixGame4.fps == 40


def test_all_scenes_switch_both_game_and_model_only_on_start(app):
    async def run():
        for index, scene in enumerate(
            ("ancient-egypt", "castle", "racing", "ancient-egypt")
        ):
            previous_game = app._game
            app.state.scene = scene
            reply = await app.start()
            assert reply.scene == scene
            if previous_game is not None:
                assert previous_game.closed
            for chunk in range(10):
                input, _ = await step(app)
                assert input.world_id == index + 1
                assert (input.start is not None) == (chunk == 0)
                if input.start is not None:
                    assert input.start.scene == scene
            assert app._engine._backend is app._engine._backends[scene]
            assert app._engine._scene == scene
            assert app.state._frame_end == 36
        assert app._engine._backends["ancient-egypt"].starts == [0, 0]
        assert app._engine._backends["castle"].closes == 1
        assert app._engine._backends["racing"].closes == 1

    asyncio.run(run())


def test_next_scene_does_not_change_live_controls_or_caption(app):
    async def run():
        await app.start()
        await step(app)
        app.state.scene = "racing"
        assert app._status().next_scene == "racing"
        assert app._status().scene == "ancient-egypt"
        app.action("jump")
        input, _ = await step(app)
        assert input.start is None and input.caption is None
        assert app._engine._scene == "ancient-egypt"
        await app.start()
        input, _ = await step(app)
        assert input.start.scene == "racing"
        with pytest.raises(CommandError, match="unavailable"):
            app.action("jump")

    asyncio.run(run())


def test_scene_is_validated_by_the_public_contract():
    contract = ModelContract.of(MatrixGame4)
    with pytest.raises(ContractError):
        contract.validate("set_scene", {"scene": "missing"})
    assert "invalid_scene" not in contract.commands["start"].description
    assert "game_unavailable" in contract.commands["start"].description


def test_model_refuses_unknown_scene_without_touching_active_stream(app):
    async def run():
        await app.start()
        input, _ = await step(app)
        backend = app._engine._backend
        with pytest.raises(InvalidStep, match="not loaded"):
            app.generate(
                replace(input, world_id=2, start=WorldStart(seed=0, scene="missing"))
            )
        assert app._engine._backend is backend
        assert backend.closes == 0 and backend.starts == [0]

    asyncio.run(run())


def test_output_rejects_a_result_from_the_wrong_scene(app):
    async def run():
        await app.start()
        input = await app.process_input()
        result = app.generate(input)
        with pytest.raises(ValueError, match="different world"):
            await app.process_output(
                StepOutcome(result=replace(result, scene="castle"), elapsed=0.2)
            )
        assert app.state._completed_chunks == 0

    asyncio.run(run())


def test_backend_preserves_stream_and_refreshes_only_changed_captions(monkeypatch):
    preprocessing = ModuleType("matrix_game.inference.preprocessing")
    preprocessing.frames_to_tensor = Mock(side_effect=lambda frames: frames)
    preprocessing.tensor_to_uint8 = Mock(side_effect=lambda pixels: pixels)
    monkeypatch.setitem(
        sys.modules, "matrix_game.inference.preprocessing", preprocessing
    )
    streams = []

    class NativeStream:
        def __init__(self, **options):
            self.options = options
            self.pushes = 0
            self.set_prompt = Mock()
            self.close = Mock()
            streams.append(self)

        def push(self, frames, *, frame_start):
            chunk = SimpleNamespace(
                pixels=frames,
                index=self.pushes,
                frame_start=frame_start,
                frame_end=frame_start + len(frames) - 1,
            )
            self.pushes += 1
            return chunk

    backend = MatrixGame4Backend.__new__(MatrixGame4Backend)
    backend.components = SimpleNamespace(
        args=SimpleNamespace(prompt="initial"),
        caption_encoder=Mock(side_effect=lambda text: f"encoded:{text}"),
        stream=NativeStream,
    )
    backend.stream = None
    backend.start(seed=9)
    first = backend.stream
    for index, text in enumerate(("initial", "changed", "changed", "second change")):
        frames = rgb(1 if index == 0 else 4)
        frame_start = 0 if index == 0 else 1 + 4 * (index - 1)
        result = backend.generate(frames, frame_start, text)
        assert result[0] is frames
        assert backend.stream is first
        assert result[1:] == (index, frame_start, frame_start + len(frames) - 1)
    assert first.options == {"seed": 9, "max_latents": 8192}
    assert backend.components.caption_encoder.call_count == 2
    assert first.set_prompt.call_count == 2
    first.close.assert_not_called()
    backend.start(seed=10)
    first.close.assert_called_once_with()
    assert backend.stream is not first and len(streams) == 2
    assert backend.prompt == "initial"
    backend.close()
    assert backend.stream is None


def test_control_schema_has_one_path_and_typed_status(app):
    contract = ModelContract.of(MatrixGame4)
    held = {
        "forward",
        "right",
        "fire",
        "ads",
        "shield",
        "crouch",
        "weapon",
        "throttle",
        "steer",
        "brake",
        "handbrake",
    }
    assert not {f"set_{name}" for name in held} & contract.commands.keys()
    assert set(contract.commands) == {
        "start",
        "stop",
        "set_paused",
        "set_controls",
        "look",
        "action",
        "set_scene",
        "set_seed",
        "set_race_mode",
        "set_route",
        "set_front_drive",
        "set_front_grip",
        "set_rear_grip",
    }
    assert contract.validate("set_controls", {"weapon": 2}).weapon == 2
    assert contract.validate("set_controls", {}).weapon is None
    for bad in (-1, 3, "rifle"):
        with pytest.raises(ContractError):
            contract.validate("set_controls", {"weapon": bad})
    schema = contract.render_schema().to_openapi()
    status = schema["components"]["schemas"]["StateUpdate"]["properties"]
    assert "controls" not in status and held <= status.keys()
    assert status["forward"]["type"] == "number"
    assert status["fire"]["type"] == "boolean"
    assert status["weapon"]["enum"] == [0, 1, 2]
    reason = schema["components"]["schemas"]["GameFinished"]["properties"]["reason"]
    assert reason["enum"] == ["native_game_ended", "rollout_limit", "native_game_error"]
    assert "`native_game_error`" in reason["description"]
    assert json.loads(json.dumps(app._status().to_wire_format()))["data"]["weapon"] == 0


def test_weapon_changes_share_controls_and_persist_when_omitted(app):
    async def run():
        await app.start()
        reply = await command(app, "set_controls", forward=1, weapon=2)
        assert reply.weapon == app.state._weapon == 2
        assert app._game.packets[-1]["forward"] == 1
        reply = await command(app, "set_controls", forward=0)
        assert reply.weapon == 2
        assert app.messages[-1] == reply
        await app.stop()
        packets = len(app.messages)
        with pytest.raises(CommandError) as error:
            await command(app, "set_controls", weapon=1)
        assert error.value.code == "not_running"
        assert len(app.messages) == packets and app.state._weapon == 2

    asyncio.run(run())


def test_new_viewer_receives_status_without_broadcast(app):
    async def run():
        client = SimpleNamespace(send=AsyncMock())
        await app.on_connected(client)
        client.send.assert_awaited_once_with(app._status())
        assert app.messages == []

    asyncio.run(run())


@pytest.mark.parametrize(
    "error",
    [
        OSError("binary missing"),
        EOFError("process exited"),
        RuntimeError("software renderer"),
        ValueError("invalid handshake"),
    ],
)
def test_game_start_failure_is_a_named_refusal_and_retry_works(app, monkeypatch, error):
    async def run():
        await app.start()
        await step(app)
        old_game, backend = app._game, app._engine._backend
        monkeypatch.setattr(matrix_game4, "NativeGame", Mock(side_effect=error))
        with pytest.raises(CommandError) as caught:
            await app.start()
        assert caught.value.code == "game_unavailable"
        assert caught.value.__cause__ is error
        assert old_game.closed and app._game is None
        assert not app.state._running and app.state._reference is None
        assert app.state._world_id == 1 and backend.closes == 1
        assert app.messages[-1].running is False
        app._engine.shutdown.assert_not_called()
        monkeypatch.setattr(matrix_game4, "NativeGame", FakeGame)
        await app.start()
        input, _ = await step(app)
        assert input.world_id == 2 and input.frame_start == 0
        assert app._engine._backend is backend

    asyncio.run(run())


def test_unknown_programming_error_is_not_disguised_as_game_failure(app):
    async def run():
        await app.start()
        app._game.next = Mock(side_effect=TypeError("programming error"))
        with pytest.raises(TypeError, match="programming error"):
            await app.process_input()
        assert not any(isinstance(message, GameFinished) for message in app.messages)
        await app.stop()

    asyncio.run(run())


def test_real_step_loop_survives_game_fault_and_another_ten_chunks(app):
    async def run():
        app._on_loop_ready()
        app._sessions_started = 1
        await app.start()
        failed, done = asyncio.Event(), asyncio.Event()
        emit = AsyncMock()
        send = app.send
        reports = []

        async def record(message):
            await send(message)
            if isinstance(message, GameFinished):
                failed.set()

        def report(value, session, rate):
            reports.append(value)
            if app.state._world_id == 1 and app.state._completed_chunks == 2:
                app._game.frame += 4
            if app.state._world_id == 2 and app.state._completed_chunks == 10:
                app._live.clear()
                done.set()

        app.send, app.emit, app._report_step = record, emit, report
        app._live.set()
        task = asyncio.create_task(app.run())
        try:
            await asyncio.wait_for(failed.wait(), timeout=5)
            assert not task.done() and not app.state._running
            assert len(reports) == emit.await_count == 2
            async with app._step_lock:
                await app.start()
            await asyncio.wait_for(done.wait(), timeout=5)
            assert not task.done()
            assert len(reports) == emit.await_count == 12
            app._engine.shutdown.assert_not_called()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await app.stop()

    asyncio.run(run())


@pytest.fixture
def native_stub(monkeypatch, tmp_path):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    calls = []
    validate = Mock()

    class Engine:
        def __init__(self, root, **kwargs):
            self.path = kwargs["log_path"]
            self.path.write_text("game log")
            self.hello = {}
            self.close = Mock()
            self.send = Mock()
            self.next = Mock(return_value=None)
            calls.append(self)

    for module_name, values in (
        ("matrix_game.server.app", {"validate_renderer": validate}),
        ("matrix_game.server.game", {"CastleGame": Engine}),
        ("matrix_game.server.racing", {"RacingGame": Engine}),
        ("matrix_game.server.reference_input", {"ReferencePreprocessor": Mock()}),
    ):
        module = ModuleType(module_name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, module_name, module)
    settings = SimpleNamespace(
        scene="castle",
        game_root=tmp_path,
        asset_root=tmp_path,
        prompt="scene",
        allow_software_rendering=False,
    )
    return settings, calls, validate


def test_native_logs_are_removed_after_each_game(native_stub, tmp_path):
    settings, calls, _ = native_stub
    for _ in range(4):
        game = NativeGame(settings, {})
        native = calls[-1]
        assert native.path.is_file() and native.path.parent.parent == tmp_path
        game.close()
        native.close.assert_called_once_with()
        assert not native.path.parent.exists()
    assert not list(tmp_path.iterdir())


def test_native_logs_are_removed_when_renderer_validation_fails(native_stub, tmp_path):
    settings, calls, validate = native_stub
    validate.side_effect = RuntimeError("software renderer")
    with pytest.raises(RuntimeError, match="software renderer"):
        NativeGame(settings, {})
    calls[0].close.assert_called_once_with()
    assert not list(tmp_path.iterdir())


def test_native_logs_are_removed_when_launch_fails(native_stub, monkeypatch, tmp_path):
    settings, _, _ = native_stub

    def fail(root, **kwargs):
        kwargs["log_path"].write_text("failed start")
        raise OSError("could not launch")

    monkeypatch.setattr(sys.modules["matrix_game.server.game"], "CastleGame", fail)
    with pytest.raises(OSError):
        NativeGame(settings, {})
    assert not list(tmp_path.iterdir())


def test_native_logs_are_removed_even_if_close_fails(native_stub, tmp_path):
    settings, calls, _ = native_stub
    game = NativeGame(settings, {})
    calls[-1].close.side_effect = RuntimeError("reader did not stop")
    with pytest.raises(RuntimeError, match="reader did not stop"):
        game.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("images", [[], [rgb(1)[0], rgb(1)[0, :2]]])
def test_malformed_capture_is_a_game_error(native_stub, images):
    settings, calls, _ = native_stub
    game = NativeGame(settings, {})
    calls[-1].next.return_value = (SimpleNamespace(images=images), [])
    try:
        with pytest.raises(NativeGameError, match="capture failed") as caught:
            game.next()
        assert isinstance(caught.value.__cause__, ValueError)
    finally:
        game.close()


@pytest.mark.parametrize(
    "operation,error",
    [
        ("next", RuntimeError("reader failed")),
        ("next", EOFError("game exited")),
        ("send", BrokenPipeError("game exited")),
    ],
)
def test_native_io_errors_are_explicit(native_stub, operation, error):
    settings, calls, _ = native_stub
    game = NativeGame(settings, {})
    getattr(calls[-1], operation).side_effect = error
    try:
        with pytest.raises(NativeGameError) as caught:
            game.next() if operation == "next" else game.send({})
        assert caught.value.__cause__ is error
    finally:
        game.close()


def test_asset_setup_does_not_create_logs_under_read_only_weights(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    (source / "matrix_game").mkdir(parents=True)
    (source / "matrix_game/scenes.py").touch()
    weights = tmp_path / "weights"
    weights.mkdir()
    weights.chmod(0o555)
    monkeypatch.setenv("MATRIX_GAME4_SOURCE_DIR", str(source))
    monkeypatch.delenv("MATRIX_GAME4_RESOURCES", raising=False)
    prepare = Mock(side_effect=lambda source, root, values, scene: scene)
    monkeypatch.setattr(matrix_game4_assets, "_prepare_scene", prepare)
    try:
        result = matrix_game4_assets.prepare(None, weights)
        assert set(result) == {"ancient-egypt", "castle", "racing"}
        assert not list(weights.iterdir())
        assert prepare.call_count == 3
    finally:
        weights.chmod(0o755)


def test_native_horizon_uses_one_shared_constant():
    assert MAX_CHUNKS == 8192
