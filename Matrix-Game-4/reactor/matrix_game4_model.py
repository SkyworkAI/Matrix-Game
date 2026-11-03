"""Runtime-independent model ownership and one native causal chunk per call."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_game4_backend import MAX_CHUNKS, MatrixGame4Backend, NativeSettings


@dataclass(frozen=True)
class WorldStart:
    """Scene and inference seed applied once when a new world is accepted."""

    seed: int
    scene: str


@dataclass(frozen=True)
class MatrixGame4Input:
    """CPU RGB uint8 (T, 480, 864, 3); T is 1 initially, then 4.

    The start value travels until the result acknowledges the world. Captions
    come from the racing engine for these exact frames. An empty frame value
    indicates the native game ended, without inventing a final video frame.
    """

    world_id: int
    start: WorldStart | None
    frames: np.ndarray | None
    frame_start: int
    caption: str | None = None


@dataclass(frozen=True)
class MatrixGame4Result:
    """Facts about the completed native step.

    Attributes:
        world_id: Identity of the world the model applied.
        chunk_index: Number of successfully generated chunks, starting at one.
        frame_start: Inclusive zero-based first frame of this chunk.
        frame_end: Inclusive last frame; frame_start - 1 for a terminal notice.
        frames: CPU RGB uint8 (T, 480, 864, 3), or None when the game ended.
        frames_wanted: Reference frames needed next; zero at completion.
        complete: The native game ended or its 8192-chunk horizon was reached.
        scene: The scene checkpoint that rendered this world.
    """

    world_id: int
    chunk_index: int
    frame_start: int
    frame_end: int
    frames: np.ndarray | None
    frames_wanted: int
    complete: bool
    scene: str


class InvalidStep(ValueError):
    """The requested world or frame range cannot continue this stream."""


class MatrixGame4Model:
    """Own a native stream until reset or a step asks for a new world ID.

    A new ID requires WorldStart and frame zero. InvalidStep rejects a missing
    start, a discontinuous frame range, an exhausted horizon, or reuse after a
    failure. Unknown backend errors propagate without claiming completion.
    """

    def __init__(self) -> None:
        self._backend: MatrixGame4Backend | None = None
        self._backends: dict[str, MatrixGame4Backend] = {}
        self._scene: str | None = None
        self._world_id: int | None = None
        self._chunk_index = 0
        self._next_frame = 0
        self._failed = False

    def load(self, settings: dict[str, NativeSettings]) -> None:
        """Load the released weights once from application-resolved settings."""
        self._backends = {
            scene: MatrixGame4Backend(native) for scene, native in settings.items()
        }

    def generate(self, input: MatrixGame4Input) -> MatrixGame4Result:
        """Run exactly one native streaming chunk or report the game's end."""
        if not self._backends:
            raise RuntimeError("load() must run before generate()")
        fresh = input.world_id != self._world_id
        if fresh and input.start is None:
            raise InvalidStep("A new world needs its seed")
        if fresh and input.start.scene not in self._backends:
            raise InvalidStep("The requested scene is not loaded")
        if not fresh and self._failed:
            raise InvalidStep("The previous chunk failed; start a fresh world")
        index = 0 if fresh else self._chunk_index
        first = 0 if fresh else self._next_frame
        if input.frame_start != first or index >= MAX_CHUNKS:
            raise InvalidStep("Nonconsecutive frames or exhausted native horizon")
        if input.frames is None:
            if fresh:
                raise InvalidStep("A fresh world needs its first game frame")
            return MatrixGame4Result(
                world_id=input.world_id,
                chunk_index=index,
                frame_start=first,
                frame_end=first - 1,
                frames=None,
                frames_wanted=0,
                complete=True,
                scene=self._scene,
            )
        count = 1 if index == 0 else 4
        if input.frames.dtype != np.uint8 or input.frames.shape != (count, 480, 864, 3):
            raise InvalidStep("Expected one initial RGB frame, then four per chunk")
        try:
            if fresh:
                if (
                    self._backend is not None
                    and self._backend is not self._backends[input.start.scene]
                ):
                    self._backend.close()
                self._backend = self._backends[input.start.scene]
                self._backend.start(input.start.seed)
                self._scene = input.start.scene
                self._world_id = input.world_id
                self._chunk_index = self._next_frame = 0
                self._failed = False
            frames, native_index, frame_start, frame_end = self._backend.generate(
                input.frames,
                input.frame_start,
                input.caption,
            )
            if (
                native_index != index
                or frame_start != first
                or frame_end != first + count - 1
                or frames.dtype != np.uint8
                or frames.shape != input.frames.shape
            ):
                raise InvalidStep("Native output does not match the input frame range")
        except BaseException:
            self._failed = True
            raise
        self._chunk_index += 1
        self._next_frame += count
        complete = self._chunk_index == MAX_CHUNKS
        return MatrixGame4Result(
            world_id=self._world_id,
            chunk_index=self._chunk_index,
            frame_start=frame_start,
            frame_end=frame_end,
            frames=frames,
            frames_wanted=0 if complete else 4,
            complete=complete,
            scene=self._scene,
        )

    def reset(self) -> None:
        """Drop the native rollout and keep its loaded model components."""
        if self._backend is not None:
            self._backend.close()
        self._world_id = None
        self._scene = None
        self._chunk_index = self._next_frame = 0
        self._failed = False
