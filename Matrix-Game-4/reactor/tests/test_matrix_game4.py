"""CPU contract tests; fake compute is not neural-generation acceptance."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome
from reactor_runtime.distributed import DistributedRunner

import matrix_game4
from matrix_game4 import MatrixGame4
from matrix_game4_backend import MatrixGame4Backend
from matrix_game4_game import ReferenceChunk
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
            "forward": state.forward,
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


@pytest.mark.asyncio
async def test_explicit_start_and_pause_refuse_without_model(app):
    with pytest.raises(ApplicationError):
        await step(app)
    await app.start()
    app.state.paused = True
    with pytest.raises(ApplicationError):
        await step(app)
    assert app._engine._backend.calls == []
    assert app._game.frame == 0


@pytest.mark.asyncio
async def test_application_hooks_against_scripted_model_half(app):
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


@pytest.mark.asyncio
async def test_ten_chunks_are_frame_paired_and_acknowledge_start(app):
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


@pytest.mark.asyncio
async def test_generate_does_not_read_application_state(app):
    await app.start()
    input = await app.process_input()
    app.state = None
    assert app.generate(input).chunk_index == 1


@pytest.mark.asyncio
async def test_failure_never_advances_completed_count(app):
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


@pytest.mark.asyncio
async def test_new_world_and_session_seed_isolation(app):
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


@pytest.mark.asyncio
async def test_pause_keeps_native_history_and_caption_transition(app):
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


@pytest.mark.asyncio
async def test_control_refusal_and_awaited_reply(app):
    with pytest.raises(CommandError) as caught:
        await app.set_controls(forward=1)
    assert caught.value.code == "not_running"
    assert app.messages == []
    await app.start()
    reply = await app.set_controls(forward=1, fire=True)
    assert reply.controls["forward"] == 1 and reply.controls["fire"]
    assert app.messages[-1] == reply
    app.look(10, -4)
    assert app._game.packets[-1]["look_x"] == 10
    app.action("jump")
    assert app._game.packets[-1]["actions"] == ["jump"]
    with pytest.raises(CommandError) as caught:
        app.action("retry")
    assert caught.value.code == "wrong_scene"


@pytest.mark.asyncio
async def test_another_viewer_leaving_does_not_reset_the_game(app):
    await app.start()
    await step(app)
    app.connected = asyncio.Event()
    app.connected.set()
    app.state.forward = 1
    game = app._game
    await app.on_disconnected(None)
    assert app.state.forward == 1
    assert app._game is game and not game.closed
    app.connected.clear()
    await app.on_disconnected(None)
    assert app.state.forward == 0
    assert app._engine._backend.closes == 0
    assert not game.closed


@pytest.mark.asyncio
async def test_native_end_is_reported_without_inventing_a_frame(app):
    await app.start()
    await step(app)
    app._game.finish = True
    _, output = await step(app)
    assert output is None
    assert not app.state._running
    assert app.state._completed_chunks == 1
    assert isinstance(app.messages[-1], GameFinished)
    assert app.messages[-1].reason == "native_game_ended"


@pytest.mark.asyncio
async def test_misaligned_reference_is_rejected_before_inference(app):
    await app.start()
    app._game.frame = 9
    with pytest.raises(ValueError, match="not consecutive"):
        await step(app)
    assert app._engine._backend.calls == []


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


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", [None, 3])
async def test_real_runtime_step_loop(app, fail_at):
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


@pytest.mark.asyncio
async def test_pause_reply_broadcasts_and_releases_held_buttons(app):
    await app.start()
    await app.set_controls(forward=1, fire=True)
    reply = await app.set_paused(True)
    assert reply.controls["paused"] is True
    assert reply.controls["forward"] == 0
    assert reply.controls["fire"] is False
    assert app.messages[-1] == reply
    await app.set_paused(False)
    assert not app.state.paused


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


@pytest.mark.asyncio
async def test_race_action_refusal_uses_active_mode_not_next_start(app):
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


@pytest.mark.asyncio
async def test_all_scenes_switch_both_game_and_model_only_on_start(app):
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


@pytest.mark.asyncio
async def test_next_scene_does_not_change_live_controls_or_caption(app):
    await app.start()
    await step(app)
    app.state.scene = "racing"
    assert app._status().controls["scene"] == "racing"
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


@pytest.mark.asyncio
async def test_unknown_scene_refused_before_closing_live_world(app):
    await app.start()
    await step(app)
    game, completed = app._game, app.state._completed_chunks
    app.state.scene = "missing"  # bypass Runtime enum validation to exercise guard
    with pytest.raises(CommandError) as caught:
        await app.start()
    assert caught.value.code == "invalid_scene"
    assert app._game is game and not game.closed
    assert app.state._completed_chunks == completed


@pytest.mark.asyncio
async def test_model_refuses_unknown_scene_without_touching_active_stream(app):
    await app.start()
    input, _ = await step(app)
    backend = app._engine._backend
    with pytest.raises(InvalidStep, match="not loaded"):
        app.generate(
            replace(input, world_id=2, start=WorldStart(seed=0, scene="missing"))
        )
    assert app._engine._backend is backend
    assert backend.closes == 0 and backend.starts == [0]


@pytest.mark.asyncio
async def test_output_rejects_a_result_from_the_wrong_scene(app):
    await app.start()
    input = await app.process_input()
    result = app.generate(input)
    with pytest.raises(ValueError, match="different world"):
        await app.process_output(
            StepOutcome(result=replace(result, scene="castle"), elapsed=0.2)
        )
    assert app.state._completed_chunks == 0


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
