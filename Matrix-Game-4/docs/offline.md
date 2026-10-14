# Local Recording and Offline Rendering

This page describes shooter recordings. See [City Racing](racing.md) for the
racing recorder, its own finalizer, and optional per-chunk caption sidecars.

Build the game and install its asset pack. No model or network is needed to record:

```bash
bash scripts/play.sh --scene ancient-egypt
```

The recorder presents 48 frames per second while advancing simulation at 2x.
Each ordinary frame advances five 1/120-second physics steps. Ten seconds of
recorded video therefore covers twenty game seconds. The camera limit is
90 degrees per simulation second, not per displayed second.

F9 starts/stops a take. Pause, lost focus, death or exit finalizes it. Resume does
not silently resume the same take. WASD/arrows move, mouse aims, left/right mouse
fire/aim, 1/2/3 select weapons, R reloads, Space jumps, Shift dashes, C crouches,
F shields, T places a mine and H places cover.

Takes are written below the game data directory's `offline-recordings/`, isolated
by scene and session. Use `--offline-dir /path/to/takes` to override the location.

| File | Contents |
| --- | --- |
| `reference.mkv` | Lossless clean 1280x720 low-poly RGBA, without UI |
| `ui.mkv` | Frame-aligned premultiplied-alpha HUD |
| `audio.wav` | 48 kHz stereo PCM, aligned to presentation time |
| `frames.jsonl` | Frame timing, applied physics inputs, camera, events |
| `inputs.jsonl` | Received input events and initial held state |
| `session.json` | Scene, fixed prompt, frame count and completion status |

```bash
python game/tools/finalize-demo.py /path/to/take-1 --check-only
matrix-game render --take /path/to/take-1 --preset quality \
  --resources resources --output outputs/take-1
python game/tools/finalize-demo.py /path/to/take-1 \
  --rendered outputs/take-1/rendered.mp4
```

The compositor checks frame counts and rates, overlays HUD after neural rendering
and restores audio. UI pixels must never enter the model. Premultiplied RGB is
not multiplied by alpha a second time.

Optional keyboard visualization:

```bash
python game/tools/add-input-overlay.py /path/to/take-1 \
  --video /path/to/demo.mp4 --output /path/to/demo-with-keys.mp4 \
  --keyboard-lead-ms 150
```

Label this as **offline rendering with a simulated input-display offset**. The
150 ms setting is not a measured model or network latency.
