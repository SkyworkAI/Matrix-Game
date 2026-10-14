# Native Games

The native C++ games run independently of the neural renderer. Each game's
desktop recorder and headless reference server share gameplay and drawing code.
Castle and Ancient Egypt use one shooter implementation with scene profiles;
City Racing is a separate implementation in `racing/`. Both reuse dependencies
under `third_party/`.

| Scene | Implementation | Game modes | Runtime assets |
| --- | --- | --- | --- |
| Medieval Castle | `src/` | Third-person shooter | `resources/game/assets/` |
| Ancient Egypt | `src/` | Third-person shooter | `resources/game/assets/` |
| City Racing | `racing/src/` | Free driving, three time-trial routes, pursuit | `resources/racing/assets/` |

All commands below run from `Matrix-Game-4/`, not from `game/` or the parent repository directory.

## Shooter Entry Points

- `src/MainLinux.cpp`: macOS/Linux desktop application and validation modes.
- `src/OfflineDemo.cpp`, `src/DemoRecorder.cpp`: 48 FPS / 2x local recording,
  clean RGB, transparent HUD, sound and frame-indexed input traces.
- `src/ReferenceServerMain.cpp`: headless reference capture and bounded IPC.
- `src/SceneConfig.cpp`: scene assets, fixed prompts, camera and visual settings.
- `src/Encounter.cpp`, `src/Collision.cpp`: actors, combat, navigation and physics.
- `src/RendererLinux.cpp`: SDL/OpenGL low-poly rendering shared by both paths.
- `src/CombatAudio.cpp`, `src/CombatUi.h`: game-state-derived audio/HUD.
- `src/Main.cpp`, `src/Renderer.cpp`: legacy Windows/Direct3D entry. This branch
  has not received a feature-parity validation; use macOS/Linux for this demo.

## Shooter Build and Resources

```bash
bash scripts/build_game.sh
bash scripts/play.sh --scene castle
# Or: bash scripts/play.sh --scene ancient-egypt
```

Build products are
written to `game/build` and `game/bin`, neither of which belongs in Git.
See [installation](../docs/installation.md) for native dependencies and
[offline recording](../docs/offline.md) for shooter recording and composition.

Assets are resolved under `resources/game/assets/` by the launch scripts.
`MATRIX_GAME_DATA_DIR` selects a different desktop resource root; the reference
server takes `--data-dir`. Asset catalogs and map binaries must come from the
same manifest revision. Do not place model checkpoints inside the game tree.

The runtime asset bundle is sufficient to build and play these two scenes.
Original authoring exports and a general-purpose map editor are not included.

## City Racing

```bash
bash scripts/build_racing.sh -DRACING_ASSET_ROOT="$PWD/resources/racing"
game/racing/bin/CitySampleBBoxGame --data-dir resources/racing --race --hud
# Or: game/racing/bin/CitySampleBBoxGame --data-dir resources/racing --pursuit --hud
```

The desktop executable supports free driving, time trials and pursuit. Race
menus include route/mission selection and handling settings. Recording, HUD,
audio and input traces are supported. Build products
are isolated in `racing/build` and `racing/bin`; runtime assets are separate from
source. Use the racing recording/finalization tools, not the shooter's 2x-time
recording schema.

- `racing/src/MainSDL.cpp`: macOS/Linux desktop application and menus.
- `racing/src/RaceSession.cpp`, `RaceRules.cpp`, `Pursuit.cpp`: game modes.
- `racing/src/World.cpp`, `Physics.cpp`: city state and vehicle physics.
- `racing/src/ReferenceServerMain.cpp`: headless reference capture and game IPC.
- `racing/src/CaptionStream.cpp`, `WorldCaptions.cpp`: tracked visible entities
  and chunk captions for dynamic prompts.
- `racing/src/OfflineRecorder.cpp`, `VehicleAudio.cpp`, `RendererHud.cpp`:
  layered recording, sound and HUD.

See [City Racing](../docs/racing.md) for controls, recording, dynamic/fixed
prompts and online inference. In the browser, select the time-trial, pursuit or
free-drive mode after starting the model service with `--scene racing`.

## Shared Model Boundary

The neural model receives clean RGB only, never the HUD, browser keyboard overlay
or audio. Those channels stay separate and are joined by frame identity at display
or offline composition. Model-generated objects do not alter collision geometry.
Keep camera, object colors, outlines, ground patterns and prompts aligned with
the released profiles when comparing model quality.

The shooter scenes have one player and two NPCs. The player has all three weapons; NPCs
use two-projectile shotgun bursts. Shared code includes damage, defeat/retry,
shields, traps, barriers, sound cues and recording. Source-level validation
modes are retained for regression tests; they are not additional model features.

Third-party source is isolated under `third_party/` with its original licenses.
