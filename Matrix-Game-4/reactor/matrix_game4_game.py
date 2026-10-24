"""Application-side native game, controls and bounded CPU reference frames."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from matrix_game4_assets import AppSettings


@dataclass(frozen=True)
class ReferenceChunk:
    frames: np.ndarray
    states: list[dict]


class NativeGameError(RuntimeError):
    """The native game could not supply valid input for another model step."""


class NativeGame:
    def __init__(self, settings: AppSettings, options: dict) -> None:
        from matrix_game.server.app import validate_renderer
        from matrix_game.server.game import CastleGame
        from matrix_game.server.racing import RacingGame
        from matrix_game.server.reference_input import ReferencePreprocessor

        self.cancel = threading.Event()
        self.game = None
        self._logs = TemporaryDirectory(prefix="matrix-game4-")
        kwargs = {
            "fps": 40,
            "log_path": Path(self._logs.name) / "game.log",
            "asset_root": settings.asset_root,
            "prepare": ReferencePreprocessor(model=False),
        }
        try:
            if settings.scene == "racing":
                self.game = RacingGame(settings.game_root, options=options, **kwargs)
            else:
                self.game = CastleGame(
                    settings.game_root,
                    scene=settings.scene,
                    expected_prompt=settings.prompt,
                    **kwargs,
                )
            validate_renderer(self.game.hello, settings.allow_software_rendering)
        except BaseException:
            self.close()
            raise

    def send(self, value: dict) -> None:
        try:
            self.game.send(value)
        except (OSError, EOFError) as error:
            raise NativeGameError("Native game control connection failed") from error

    def next(self) -> ReferenceChunk | None:
        try:
            value = self.game.next(self.cancel)
            if value is None:
                return None
            reference, states = value
            return ReferenceChunk(
                frames=np.stack([np.asarray(frame) for frame in reference.images]),
                states=states,
            )
        except (OSError, EOFError, RuntimeError, ValueError) as error:
            raise NativeGameError("Native game capture failed") from error

    def close(self) -> None:
        self.cancel.set()
        try:
            if self.game is not None:
                self.game.close()
        finally:
            self._logs.cleanup()


def control_packet(state, scene: str, sequence: int, **impulses) -> dict:
    if scene == "racing":
        from matrix_game.server.racing import controls

        return controls(
            dict(
                sequence=sequence,
                drive=[
                    state._throttle,
                    state._steer,
                    state._brake,
                    int(state._handbrake),
                ],
                **impulses,
            )
        )
    from matrix_game.server.app import controls

    return controls(
        dict(
            sequence=sequence,
            forward=state._forward,
            right=state._right,
            fire=state._fire,
            ads=state._ads,
            shield=state._shield,
            crouch=state._crouch,
            weapon=state._weapon,
            **impulses,
        )
    )
