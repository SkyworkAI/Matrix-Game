"""Matrix-Game 4.0: native game controls and frame-paired neural rendering."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    TrackPayload,
    connected,
    disconnected,
    event,
    get_weights_path,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner

from matrix_game4_assets import prepare
from matrix_game4_game import NativeGame, NativeGameError, control_packet
from matrix_game4_model import (
    MatrixGame4Input,
    MatrixGame4Model,
    MatrixGame4Result,
    WorldStart,
)
from matrix_game4_types import (
    GameFinished,
    MatrixGame4Output,
    MatrixGame4State,
    StateUpdate,
)

logger = logging.getLogger(__name__)


class MatrixGame4(ReactorApp):
    """Play a native Matrix-Game scene with its draft and neural render side by side."""

    state: MatrixGame4State
    buffer_size = 4
    # Native game physics/capture advances at 40 frames per simulated second.
    # Faster inference must not compress those timestamps into a short burst.
    fps = 40

    def __init__(self) -> None:
        super().__init__()
        self._settings = {}
        self._engine = None
        self._game = None

    def load(self, config_path: Path | None) -> None:
        self._settings = prepare(config_path, get_weights_path())
        self._engine = DistributedRunner(
            MatrixGame4Model,
            world_size=1,
            start_timeout=600,
            call_timeout=300,
            load_kwargs={
                "settings": {
                    scene: settings.native for scene, settings in self._settings.items()
                }
            },
        )
        self._engine.start()

    async def process_input(self) -> MatrixGame4Input:
        state = self.state
        if not state._running:
            raise ApplicationError("Press start to begin a game")
        if state.paused:
            raise ApplicationError("paused")
        settings = self._settings[state._active_scene]
        caption = None
        try:
            self._send_controls()
            reference = await asyncio.to_thread(self._game.next)
            if reference is not None:
                expected = list(
                    range(
                        state._frame_end + 1,
                        state._frame_end + 1 + state._frames_wanted,
                    )
                )
                if (
                    [s.get("frame") for s in reference.states] != expected
                    or reference.frames.dtype != np.uint8
                    or reference.frames.shape != (state._frames_wanted, 480, 864, 3)
                ):
                    raise NativeGameError("Native reference frames are misaligned")
                if (
                    settings.prompt_mode == "dynamic"
                    and state._completed_chunks % settings.prompt_interval_chunks == 0
                ):
                    caption = reference.states[-1].get("chunk_prompt")
                    if not isinstance(caption, str) or not caption.strip():
                        raise NativeGameError("Native racing caption is missing")
        except NativeGameError as error:
            logger.warning("Ending game after an input failure", exc_info=True)
            await self._close_game()
            self._engine.reset()
            state._running = False
            state._reference = None
            self._release_controls()
            self.output.flush()
            await self.send(self._status())
            await self.send(
                GameFinished(world_id=state._world_id, reason="native_game_error")
            )
            raise ApplicationError("Game input failed; start a new game") from error
        state._reference = reference
        return MatrixGame4Input(
            world_id=state._world_id,
            start=WorldStart(seed=state._active_seed, scene=state._active_scene)
            if state._applied_world_id != state._world_id
            else None,
            frames=reference.frames if reference is not None else None,
            frame_start=state._frame_end + 1,
            caption=caption,
        )

    def generate(self, input: MatrixGame4Input) -> MatrixGame4Result:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> MatrixGame4Output | None:
        if outcome.error is not None:
            await self._close_game()
            self._engine.shutdown()
            # A failed native inference call may have mutated KV/decoder state.
            # It must not be retried as a successful or silently reset chunk.
            raise outcome.error
        result: MatrixGame4Result = outcome.result
        state = self.state
        if result.world_id != state._world_id or result.scene != state._active_scene:
            raise ValueError("Result belongs to a different world")
        reference = state._reference
        output = None
        if result.frames is not None:
            if (
                reference is None
                or result.frames.shape != reference.frames.shape
                or result.frame_start != reference.states[0]["frame"]
                or result.frame_end != reference.states[-1]["frame"]
                or result.chunk_index != state._completed_chunks + 1
            ):
                raise ValueError("Generated video does not match its reference chunk")
            # Both halves share one encoded frame and presentation timestamp.
            paired = np.concatenate((reference.frames, result.frames), axis=2)
            metadata = [
                {
                    "world_id": result.world_id,
                    "scene": result.scene,
                    "chunk": result.chunk_index,
                    "game": s,
                }
                for s in reference.states
            ]
            output = MatrixGame4Output(
                main_video=TrackPayload(paired, metadata=metadata)
            )
            state._last_seconds = outcome.elapsed
        state._reference = None
        state._applied_world_id = result.world_id
        state._completed_chunks = result.chunk_index
        state._frame_end = result.frame_end
        state._frames_wanted = result.frames_wanted
        if result.complete:
            await self._close_game()
            self._engine.reset()
            state._running = False
        await self.send(self._status())
        if result.complete:
            await self.send(
                GameFinished(
                    world_id=result.world_id,
                    reason="native_game_ended"
                    if result.frames is None
                    else "rollout_limit",
                )
            )
        return output

    @event(
        name="start",
        description="Start a fresh native game using the selected scene, seed and racing settings. Clears the previous video and starts generation. Fails with `game_unavailable` when the native game cannot launch.",
    )
    async def start(self) -> StateUpdate:
        state = self.state
        await self._close_game()
        state._running = False
        state._reference = None
        self._release_controls()
        self.output.flush()
        try:
            self._game = await asyncio.to_thread(
                NativeGame,
                self._settings[state.scene],
                {
                    "mode": state.race_mode,
                    "route": state.route,
                    "front_drive": state.front_drive,
                    "front_grip": state.front_grip,
                    "rear_grip": state.rear_grip,
                },
            )
        except (OSError, EOFError, RuntimeError, ValueError) as error:
            logger.warning("Native game could not start", exc_info=True)
            self._engine.reset()
            await self.send(self._status())
            raise CommandError(
                "game_unavailable", "The game could not start. Please try again."
            ) from error
        state._world_id += 1
        state._active_seed = state.seed
        state._active_scene = state.scene
        state._active_race_mode = state.race_mode
        state._completed_chunks = 0
        state._frame_end = -1
        state._frames_wanted = 1
        state._last_seconds = None
        state._reference = None
        state.paused = False
        self._release_controls()
        state._running = True
        update = self._status()
        await self.send(update)
        return update

    @event(
        name="stop",
        description="Stop the active game and release its rollout state; keep the model loaded.",
    )
    async def stop(self) -> StateUpdate:
        await self._close_game()
        self._engine.reset()
        self.state._running = False
        self.state._reference = None
        self._release_controls()
        self.output.flush()
        update = self._status()
        await self.send(update)
        return update

    @event(
        name="set_paused",
        description="Hold or resume the current game and video without starting a new world. Fails with `not_running` when no game is active.",
    )
    async def set_paused(self, paused: bool) -> StateUpdate:
        self._require_game()
        self.state.paused = paused
        if paused:
            self._release_controls()
            self._send_controls()
        update = self._status()
        await self.send(update)
        return update

    @event(
        name="set_controls",
        description="Atomically update held movement and buttons. Fails with `not_running` before start or after stop.",
    )
    async def set_controls(
        self,
        forward: float = InputField(default=0, ge=-1, le=1),
        right: float = InputField(default=0, ge=-1, le=1),
        fire: bool = InputField(default=False),
        ads: bool = InputField(default=False),
        shield: bool = InputField(default=False),
        crouch: bool = InputField(default=False),
        throttle: float = InputField(default=0, ge=-1, le=1),
        steer: float = InputField(default=0, ge=-1, le=1),
        brake: float = InputField(default=0, ge=0, le=1),
        handbrake: bool = InputField(default=False),
        weapon: int | None = InputField(
            default=None,
            ge=0,
            le=2,
            description="Shooter weapon: 0 pistol, 1 rifle, 2 shotgun. Omit to keep the current weapon.",
        ),
    ) -> StateUpdate:
        self._require_game()
        state = self.state
        if weapon is not None:
            state._weapon = weapon
        state._forward, state._right = forward, right
        state._fire, state._ads, state._shield, state._crouch = (
            fire,
            ads,
            shield,
            crouch,
        )
        state._throttle, state._steer, state._brake, state._handbrake = (
            throttle,
            steer,
            brake,
            handbrake,
        )
        self._send_controls()
        update = self._status()
        await self.send(update)
        return update

    @event(
        name="look",
        description="Apply a shooter mouse impulse at the next native physics tick. Fails with `not_running` without a game, or `wrong_scene` in racing.",
    )
    def look(
        self,
        dx: float = InputField(ge=-2000, le=2000),
        dy: float = InputField(ge=-2000, le=2000),
    ) -> None:
        self._require_game()
        if self.state._active_scene == "racing":
            raise CommandError(
                "wrong_scene", "Mouse look is available in the shooter scenes"
            )
        self._send_controls(look_x=dx, look_y=dy)

    @event(
        name="action",
        description="Send one native action. Fails with `not_running` without a game, `wrong_scene` for an action unavailable in the loaded scene, or `wrong_mode` for race actions in free drive and recovery in pursuit.",
    )
    def action(
        self,
        action: Literal[
            "jump",
            "dash",
            "reload",
            "trap",
            "cover",
            "look_reset",
            "pause",
            "resume",
            "retry",
            "recover",
        ],
    ) -> None:
        self._require_game()
        racing = self.state._active_scene == "racing"
        race_action = action in ("pause", "resume", "retry", "recover")
        if racing != race_action:
            raise CommandError(
                "wrong_scene", "This action is unavailable in the loaded scene"
            )
        if racing:
            if self.state._active_race_mode == "drive":
                raise CommandError(
                    "wrong_mode", "Race actions require time trial or pursuit mode"
                )
            if action == "recover" and self.state._active_race_mode == "pursuit":
                raise CommandError(
                    "wrong_mode", "Recovery is disabled in native pursuit mode"
                )
            self._send_controls(action=action)
        elif action == "look_reset":
            self._send_controls(look_reset=True)
        else:
            self._send_controls(actions=[action])

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        await client.send(self._status())

    @disconnected
    async def on_disconnected(self, client: ClientInfo) -> None:
        if not self.connected.is_set() and self.state is not None:
            self._release_controls()
            if self._game is not None:
                self._send_controls()
            await self.send(self._status())

    @session_ended
    async def on_session_ended(self) -> None:
        await self._close_game()
        if self._engine.healthy:
            self._engine.reset()
        else:
            self._engine.shutdown()
        self.output.flush()

    async def _close_game(self) -> None:
        if self._game is not None:
            game, self._game = self._game, None
            await asyncio.to_thread(game.close)

    def _require_game(self) -> None:
        if not self.state._running:
            raise CommandError("not_running", "Start a game first")

    def _send_controls(self, **impulses) -> None:
        self.state._sequence += 1
        self._game.send(
            control_packet(
                self.state, self.state._active_scene, self.state._sequence, **impulses
            )
        )

    def _release_controls(self) -> None:
        state = self.state
        state._forward = state._right = state._throttle = state._steer = (
            state._brake
        ) = 0.0
        state._fire = state._ads = state._shield = state._crouch = state._handbrake = (
            False
        )

    def _status(self) -> StateUpdate:
        state = self.state
        return StateUpdate(
            scene=state._active_scene,
            running=state._running,
            world_id=state._world_id,
            completed_chunks=state._completed_chunks,
            frame_end=state._frame_end,
            last_chunk_seconds=state._last_seconds,
            paused=state.paused,
            next_scene=state.scene,
            seed=state.seed,
            forward=state._forward,
            right=state._right,
            fire=state._fire,
            ads=state._ads,
            shield=state._shield,
            crouch=state._crouch,
            weapon=state._weapon,
            throttle=state._throttle,
            steer=state._steer,
            brake=state._brake,
            handbrake=state._handbrake,
            race_mode=state.race_mode,
            active_race_mode=state._active_race_mode,
            route=state.route,
            front_drive=state.front_drive,
            front_grip=state.front_grip,
            rear_grip=state.rear_grip,
        )
