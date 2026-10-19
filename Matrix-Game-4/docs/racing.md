# City Racing

The native racing game supports free driving, three time-trial routes, and a
single-player pursuit. The local desktop executable includes layered offline
recording. Runtime map/audio/caption assets are downloaded separately from code.

## Build And Play

Linux and macOS require CMake, a C++20 compiler, SDL2, OpenGL, zlib, pkg-config,
and FFmpeg development libraries (avcodec, avformat, avutil, swscale).

```bash
bash scripts/build_racing.sh -DRACING_ASSET_ROOT="$PWD/resources/racing"
game/racing/bin/CitySampleBBoxGame --data-dir resources/racing --race --hud
game/racing/bin/CitySampleBBoxGame --data-dir resources/racing --pursuit --hud
```

W/S accelerate/reverse, A/D steer, Space applies the handbrake. The browser also
accepts arrow keys; P pauses a race, Enter resumes, and R requests recovery in a
time trial. Pursuit recovery is disabled. Stop/Play starts a new game and model
session. Handling defaults are front drive 0.7, front grip 1.15, rear grip 0.9.

For a local 48 FPS recording, add `--fps 48 --physics-hz 480 --offline-record captures`.
F9 toggles recording. The take contains clean reference video, alpha UI, audio,
input events and per-frame telemetry. Do not feed the UI track into the model.

```bash
python game/racing/tools/finalize-offline-demo.py captures/take-001 --check-only
python game/racing/tools/finalize-offline-demo.py captures/take-001 \
  --rendered rendered-take/rendered.mp4
```

The racing finalizer validates its own recording schema; do not use the shooter
48 FPS / 2x validator on racing takes. `compose-demo.py` in the same directory
provides optional HUD relayout and input visualization (`--help` for options).

## Online

Use the MediaMTX and SSH setup in [online.md](online.md), then:

```bash
CUDA_VISIBLE_DEVICES=0 matrix-game serve --scene racing --resources resources \
  --preset realtime --prompt-mode dynamic --prompt-interval-chunks 1 \
  --fps 24 --profile-chunks
```

`--prompt-mode fixed` uses the profile caption, optionally replaced by `--prompt`.
Dynamic mode uses visibility/depth-tested engine entities from the exact input
chunk, not object class IDs or a reused opening caption. Identical text is cached;
changed text is encoded with resident T5 and invalidates cross-attention KV only.
`--t5-runtime graph` retains native 512-token padding. The bounded text cache has
eight entries. Updating captions does not reset the visual KV or VAE history.

Racing keeps its sink 11 positions ahead of the current block once the 21-latent
window is full (future sink RoPE, `matrix_game/inference/sink_policy.py`). The
configured image is encoded with
the selected VAE, decoded as output frame zero, and written to clean KV without
denoising. Low-poly VAE history remains independent. Later blocks contain one
latent / four RGB frames. The realtime preset uses TAEHV for the sink and both
streaming codecs, one denoising step, ring/fused DiT and VAE graphs, on one GPU.
No SP launch mode is provided. Shooter profiles retain their independent KV-only
sink and fixed-relative sink RoPE; these are not interchangeable policies.

## Offline And Validation

```bash
matrix-game render --scene racing --take captures/take-001 \
  --resources resources --preset realtime --output rendered-take
```

For offline dynamic conditioning, pass `--prompt-mode dynamic --captions captions.jsonl`.
Each JSONL record has `frame_start`, `frame_end` (inclusive, zero-based), and
`prompt`. Ranges must cover frame 0, then consecutive groups of four frames; the
last range may be shorter. A plain RGB video does not supply engine captions.

Online `model_fps` includes prompt updates and post-reference processing;
`online_fps` additionally includes waiting for live game frames and is measured
after 25 startup chunks. Browser FPS measures actual presented video. None of
these alone is an action-to-photon latency measurement. Profile reports retain
bounded stage timings. `prompt_submit_ms` is host tokenization/launch time;
asynchronous GPU T5 work is included in the full processing/cycle time, not
necessarily in that host timer. Test moving scenes, changing captions, both game modes,
cache rollover, and stop/restart before publishing benchmark numbers.
