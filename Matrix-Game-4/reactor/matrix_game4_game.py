"""Application-side native game, controls and bounded CPU reference frames."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

import numpy as np

from matrix_game4_assets import AppSettings


@dataclass(frozen=True)
class ReferenceChunk:
    frames: np.ndarray
    states: list[dict]


class NativeGame:
    def __init__(self, settings: AppSettings, options: dict) -> None:
        from matrix_game.server.app import validate_renderer
        from matrix_game.server.game import CastleGame
        from matrix_game.server.racing import RacingGame
        from matrix_game.server.reference_input import ReferencePreprocessor

        self.cancel = threading.Event()
        self.game = None
        kwargs = {
            "fps": 40,
            "log_path": settings.log_root / f"game-{uuid.uuid4().hex}.log",
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
        self.game.send(value)

    def next(self) -> ReferenceChunk | None:
        value = self.game.next(self.cancel)
        if value is None:
            return None
        reference, states = value
        return ReferenceChunk(
            frames=np.stack([np.asarray(frame) for frame in reference.images]),
            states=states,
        )

    def close(self) -> None:
        self.cancel.set()
        if self.game is not None:
            self.game.close()


def control_packet(state, scene: str, sequence: int, **impulses) -> dict:
    if scene == "racing":
        from matrix_game.server.racing import controls

        return controls(
            dict(
                sequence=sequence,
                drive=[state.throttle, state.steer, state.brake, int(state.handbrake)],
                **impulses,
            )
        )
    from matrix_game.server.app import controls

    return controls(
        dict(
            sequence=sequence,
            forward=state.forward,
            right=state.right,
            fire=state.fire,
            ads=state.ads,
            shield=state.shield,
            crouch=state.crouch,
            weapon=state.weapon,
            **impulses,
        )
    )
