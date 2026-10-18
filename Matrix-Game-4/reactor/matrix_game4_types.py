"""Client controls and messages; native frame IDs travel on the video track."""

from __future__ import annotations

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
    forward: float = InputField(
        default=0.0,
        ge=-1,
        le=1,
        description="Shooter movement: -1 backward, 1 forward; held until changed.",
    )
    right: float = InputField(
        default=0.0,
        ge=-1,
        le=1,
        description="Shooter movement: -1 left, 1 right; held until changed.",
    )
    fire: bool = InputField(default=False, description="Hold the shooter trigger.")
    ads: bool = InputField(default=False, description="Hold shooter aim-down-sights.")
    shield: bool = InputField(default=False, description="Hold the shooter's shield.")
    crouch: bool = InputField(default=False, description="Hold shooter crouch.")
    weapon: int = InputField(
        default=0,
        ge=0,
        le=2,
        description="Shooter weapon: 0 pistol, 1 rifle, 2 shotgun.",
    )
    throttle: float = InputField(
        default=0.0,
        ge=-1,
        le=1,
        description="Racing throttle: -1 reverse, 1 accelerate.",
    )
    steer: float = InputField(
        default=0.0, ge=-1, le=1, description="Racing steering: -1 left, 1 right."
    )
    brake: float = InputField(
        default=0.0, ge=0, le=1, description="Racing service brake."
    )
    handbrake: bool = InputField(
        default=False, description="Hold the racing handbrake."
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
        description="Scene of the displayed world; null before the first start. The next-start selection is in controls.scene."
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
    controls: dict = MessageField(
        description="All current controls and next-start settings. Control changes affect future captured frames; buffered frames keep playing."
    )


class GameFinished(ModelMessage):
    world_id: int = MessageField(description="The game that finished.")
    reason: str = MessageField(description="native_game_ended or rollout_limit.")


class MatrixGame4Output(Output):
    main_video: Video
