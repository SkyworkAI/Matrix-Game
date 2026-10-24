"""Client controls and messages; native frame IDs travel on the video track."""

from __future__ import annotations

from typing import Literal

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

from matrix_game4_game import ReferenceChunk


class MatrixGame4State(InputState):
    scene: str = InputField(
        default="ancient-egypt",
        choices=["ancient-egypt", "castle", "racing"],
        description="Scene for the next start. Changing it does not switch an active game.",
    )
    paused: bool = InputField(
        default=False,
        description="Hold generation and game capture without starting a new world.",
    )
    seed: int = InputField(
        default=0, ge=0, le=2147483647, description="Inference seed for the next start."
    )
    race_mode: str = InputField(
        default="time_trial",
        choices=["drive", "time_trial", "pursuit"],
        description="Racing mode for the next start.",
    )
    route: str = InputField(
        default="downtown",
        choices=["downtown", "avenue", "skyway"],
        description="Route for the next start; pursuit uses avenue.",
    )
    front_drive: float = InputField(
        default=0.7,
        ge=0,
        le=1,
        description="Front-wheel drive fraction for the next racing start.",
    )
    front_grip: float = InputField(
        default=1.15,
        ge=0.9,
        le=1.2,
        description="Front tire grip for the next racing start.",
    )
    rear_grip: float = InputField(
        default=0.9,
        ge=0.9,
        le=1.2,
        description="Rear tire grip for the next racing start.",
    )

    _forward: float = 0.0
    _right: float = 0.0
    _fire: bool = False
    _ads: bool = False
    _shield: bool = False
    _crouch: bool = False
    _weapon: int = 0
    _throttle: float = 0.0
    _steer: float = 0.0
    _brake: float = 0.0
    _handbrake: bool = False
    _running: bool = False
    _world_id: int = 0
    _applied_world_id: int | None = None
    _active_seed: int = 0
    _active_scene: str | None = None
    _active_race_mode: str = "time_trial"
    _completed_chunks: int = 0
    _frame_end: int = -1
    _frames_wanted: int = 1
    _last_seconds: float | None = None
    _reference: ReferenceChunk | None = None
    _sequence: int = 0


class StateUpdate(ModelMessage):
    scene: str | None = MessageField(
        description="Scene of the displayed world; null before the first start."
    )
    running: bool = MessageField(description="Whether a game is active.")
    world_id: int = MessageField(description="Current game identity; changes on start.")
    completed_chunks: int = MessageField(
        description="Successfully rendered chunks in this game."
    )
    frame_end: int = MessageField(
        description="Zero-based last generated frame, or -1 before generation."
    )
    last_chunk_seconds: float | None = MessageField(
        description="Generation time of the latest completed chunk; null before generation."
    )
    next_scene: Literal["ancient-egypt", "castle", "racing"] = MessageField(
        description="Selected scene for the next start."
    )
    seed: int = MessageField(description="Inference seed for the next start.")
    paused: bool = MessageField(
        description="Whether game capture and generation are held."
    )
    forward: float = MessageField(description="Held shooter forward/backward movement.")
    right: float = MessageField(description="Held shooter right/left movement.")
    fire: bool = MessageField(description="Whether the shooter trigger is held.")
    ads: bool = MessageField(description="Whether shooter aim is held.")
    shield: bool = MessageField(description="Whether the shield is held.")
    crouch: bool = MessageField(description="Whether crouch is held.")
    weapon: Literal[0, 1, 2] = MessageField(
        description="Selected shooter weapon: 0 pistol, 1 rifle, 2 shotgun."
    )
    throttle: float = MessageField(description="Held racing throttle/reverse input.")
    steer: float = MessageField(description="Held racing steering input.")
    brake: float = MessageField(description="Held service brake input.")
    handbrake: bool = MessageField(description="Whether the handbrake is held.")
    race_mode: Literal["drive", "time_trial", "pursuit"] = MessageField(
        description="Racing mode for the next start."
    )
    active_race_mode: Literal["drive", "time_trial", "pursuit"] = MessageField(
        description="Racing mode applied to the current world."
    )
    route: Literal["downtown", "avenue", "skyway"] = MessageField(
        description="Selected route for the next start; pursuit uses avenue."
    )
    front_drive: float = MessageField(
        description="Next-start front-wheel drive fraction."
    )
    front_grip: float = MessageField(description="Next-start front tire grip.")
    rear_grip: float = MessageField(description="Next-start rear tire grip.")


class GameFinished(ModelMessage):
    world_id: int = MessageField(description="The game that finished.")
    reason: Literal["native_game_ended", "rollout_limit", "native_game_error"] = (
        MessageField(
            description="`native_game_ended`, `rollout_limit`, or `native_game_error`. Start a new game to continue."
        )
    )


class MatrixGame4Output(Output):
    main_video: Video
