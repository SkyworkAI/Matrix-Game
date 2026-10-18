"""Resolve native source, release assets and inference arguments in the application."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from matrix_game4_backend import NativeSettings

RESOURCE_REVISION = "4419accdb7bb7881234c5914de8cd915266e0325"


@dataclass(frozen=True)
class AppSettings:
    native: NativeSettings
    scene: str
    prompt: str
    prompt_mode: str
    prompt_interval_chunks: int
    game_root: Path
    asset_root: Path
    log_root: Path
    allow_software_rendering: bool


def prepare(config_path: Path | None, weights_root: Path) -> dict[str, AppSettings]:
    values = yaml.safe_load(config_path.read_text()) if config_path else {}
    values = values or {}
    parent = config_path.parent if config_path else Path(__file__).parent
    sibling = Path(__file__).resolve().parent.parent
    default_source = (
        sibling if (sibling / "matrix_game").is_dir() else Path("/opt/matrix-game4")
    )
    source = Path(
        os.environ.get(
            "MATRIX_GAME4_SOURCE_DIR", values.get("source_dir", default_source)
        )
    )
    source = (
        (parent / source).resolve() if not source.is_absolute() else source.resolve()
    )
    if not (source / "matrix_game/scenes.py").is_file():
        raise FileNotFoundError(
            "Matrix-Game-4 source is missing; build the Reactor image first"
        )
    sys.path.insert(0, str(source))
    root = Path(
        os.environ.get("MATRIX_GAME4_RESOURCES", weights_root / "resources")
    ).resolve()
    logs = Path(weights_root) / "runtime"
    logs.mkdir(parents=True, exist_ok=True)
    return {
        scene: _prepare_scene(source, root, logs, values, scene)
        for scene in ("ancient-egypt", "castle", "racing")
    }


def _prepare_scene(source, root, logs, values, scene) -> AppSettings:
    from matrix_game.resources import main as resources
    from matrix_game.scenes import add_model_arguments, resolve_model_arguments

    for manifest in ("game-assets", "models"):
        resources(
            [
                "download",
                "--manifest",
                str(source / "releases" / f"{manifest}.json"),
                "--repo-id",
                "Skywork/Matrix-Game-4.0",
                "--repo-type",
                "model",
                "--revision",
                RESOURCE_REVISION,
                "--scene",
                scene,
                "--resources",
                str(root),
            ]
        )
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    args = parser.parse_args(
        [
            "--scene",
            scene,
            "--resources",
            str(root),
            "--preset",
            values.get("preset", "realtime"),
            "--prompt-mode",
            values.get("prompt_mode", "fixed") if scene == "racing" else "fixed",
            "--prompt-interval-chunks",
            str(values.get("prompt_interval_chunks", 1)),
        ]
    )
    resolve_model_arguments(args)
    game_root = source / "game"
    executable = game_root / (
        "racing/bin/CitySampleReferenceServer"
        if scene == "racing"
        else "bin/CastleReferenceServer"
    )
    if not executable.is_file():
        raise FileNotFoundError(f"Native reference game must be built: {executable}")
    return AppSettings(
        native=NativeSettings(source_dir=source, arguments=vars(args)),
        scene=scene,
        prompt=args.prompt,
        prompt_mode=args.prompt_mode,
        prompt_interval_chunks=args.prompt_interval_chunks,
        game_root=game_root,
        asset_root=args.asset_root,
        log_root=logs,
        allow_software_rendering=bool(values.get("allow_software_rendering", False)),
    )
