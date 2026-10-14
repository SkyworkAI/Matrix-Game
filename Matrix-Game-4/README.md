# Matrix-Game 4.0

Streaming neural rendering for executable low-poly games.

[Project website](https://matrix-game-v4-0.github.io/page/)
| [Models and game assets](https://huggingface.co/Skywork/Matrix-Game-4.0)

The game engine owns gameplay, physics and input. A causal video model renders
the clean low-poly RGB stream; HUD and sound remain frame-aligned presentation
layers outside the model. This release includes **Medieval Castle**, **Ancient
Egypt**, and **City Racing**, with offline rendering and a single-user WebRTC
browser demo. Racing includes time trials and pursuit, plus fixed/dynamic prompts.

**2026-10-14:** We plan to release optimized inference code for NVIDIA GeForce
RTX 50-series GPUs later this week.

## Requirements

- Model inference: Linux, NVIDIA GPU, Python 3.12. This release is tested on
  A100 80 GB with PyTorch 2.12.0/CUDA 13.2. RTX 50-series optimizations are planned
  as noted above.
- Browser demo: the same Linux host also builds/runs the game and GStreamer.
  Safari and Chrome are the desktop client targets. No GPU is needed on the client.
- Standalone game/recording: macOS or Linux, CMake, C++20, SDL2 and FFmpeg libraries.
  The legacy Windows entry is retained but has not been validated for feature parity.
- All runtime assets and model weights are downloaded separately from source.
  Allow additional disk space for the Hub cache as well as the installed copy.

## Install

Clone the repository and enter the v4 project directory. All commands in this
guide run from `Matrix-Game-4/`, not the parent repository directory.

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git
cd Matrix-Game/Matrix-Game-4
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e '.[inference,server,download,test]'
```

The optimized path requires matching PyTorch/Triton and a compatible NVIDIA
driver. FlashAttention is optional; SDPA is the tested source deployment path.
See [installation](docs/installation.md) for native and media dependencies.

## Resources

Keep downloads outside Git, using this layout by default:

```text
resources/
  base/                        # Shared umT5, tokenizer, Wan VAE, TAEHV
  models/castle/                # model.safetensors, sink.png
  models/ancient-egypt/          # model.safetensors, sink.png
  models/racing/                # model.safetensors, sink.png
  game/assets/                  # Shooter assets and per-scene maps
  racing/assets/                # City map, captions, routes and vehicle audio
```

Model weights, the fast codec, sink images and game assets are hosted on
[Hugging Face](https://huggingface.co/Skywork/Matrix-Game-4.0). Shared T5/tokenizer
and Wan VAE files are downloaded automatically from a pinned official base-model
revision rather than duplicated in this repository. Download one scene and its
shared dependencies:

```bash
matrix-game download --manifest releases/game-assets.json \
  --repo-id Skywork/Matrix-Game-4.0 --repo-type model \
  --revision 4419accdb7bb7881234c5914de8cd915266e0325 \
  --scene ancient-egypt --resources resources

matrix-game download --manifest releases/models.json \
  --repo-id Skywork/Matrix-Game-4.0 --repo-type model \
  --revision 4419accdb7bb7881234c5914de8cd915266e0325 \
  --scene ancient-egypt --resources resources
```

Replace `ancient-egypt` with `castle`, `racing` or `all` as needed. The downloader
verifies file sizes and SHA256 against the pinned manifests. To recheck an
installed scene:

```bash
matrix-game verify --manifest releases/game-assets.json \
  --scene ancient-egypt --resources resources
matrix-game verify --manifest releases/models.json \
  --scene ancient-egypt --resources resources
```

See [resource distributions](releases/README.md) for file sizes and upstream
dependencies.

## Offline Rendering

Start with the fast path: TAEHV encoding/decoding, one denoising step, a fixed
scene prompt and the accelerated DiT runtime. Select it explicitly with
`--preset realtime`, which supports both offline and online rendering:

```bash
matrix-game render --scene ancient-egypt --input lowpoly.mp4 \
  --resources resources --preset realtime --prompt-mode fixed \
  --output outputs/egypt
```

Output is `rendered.mp4` plus a configuration report. FPS and frame count are
preserved; only the final incomplete four-frame block is padded internally, then
trimmed from the output.

To render a local shooter recording with the scene read from its manifest:

```bash
matrix-game render --take /path/to/take-1 --resources resources \
  --preset realtime --prompt-mode fixed --output outputs/take-1
python game/tools/finalize-demo.py /path/to/take-1 \
  --rendered outputs/take-1/rendered.mp4
```

For an alternative configuration, replace `--preset realtime` with
`--preset quality`: Wan VAE encoding/decoding, two denoising steps (`1000,500`,
including startup) and native DiT. This is slower and remains the CLI default
when `--preset` is omitted.

See [offline recording](docs/offline.md) for clean RGB, HUD, audio and input traces.

## Play and Record Locally

Run the games without the neural renderer to test gameplay, record model inputs
or customize the source with a coding agent. Edit terrain, object placement, NPC
behavior and game rules, then test those changes locally without loading model
weights or installing Python model dependencies.

```bash
bash scripts/build_game.sh
bash scripts/play.sh --scene ancient-egypt
```

For the two shooter scenes, `play.sh` starts the **48 FPS / 2x simulation-time**
local demo mode without neural rendering or simulated network delay. F9 toggles
recording; Escape opens the menu. Use `--scene castle` for the other map.
`MATRIX_GAME_DATA_DIR` overrides `resources/game`.

For racing, use `bash scripts/build_racing.sh`, then follow
[City Racing](docs/racing.md) for local play/recording and the browser modes.
Racing's 48 FPS recording mode uses normal 1x simulation time. The browser demo
also does not enable the shooters' local 2x mode: each captured frame advances
simulation by `1 / --fps` seconds.

## Browser Demo

Build the native game and install the media dependencies first. Run MediaMTX in
one terminal and the model service in another:

```bash
mediamtx configs/mediamtx.yml
```

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh \
  --scene ancient-egypt --resources resources --preset realtime --fps 40
```

Open **http://127.0.0.1:8025/** on the server, or forward the required ports through
SSH as described in [online deployment](docs/online.md). Add `--reference-only`
to test controls, HUD, audio and transport without loading a model.

The `realtime` preset combines TAEHV, one-step denoising, ring/fused DiT and CUDA
graph replay. `--fps 40` is a **target capture cadence, not a throughput or latency
guarantee**. Video encoding, network transport, buffering and browser decoding
add latency beyond model inference. Contributions that reduce streaming latency
are welcome.

### Scene Behavior

Castle and Ancient Egypt use fixed scene prompts. Racing supports both fixed
prompts and dynamic, chunk-aligned descriptions supplied by the game engine.
Dynamic mode updates text conditioning as the scene changes; unchanged captions
are cached, but frequent T5 updates reduce throughput.

The scene profiles also select the reference-image behavior automatically:
shooters use the reference as visual context and generate the first visible
frame; racing displays its supplied reference as frame zero, then generates the
following frames. See [City Racing](docs/racing.md) and
[architecture](docs/architecture.md) for prompt and temporal-position details.

## A100 Performance

Reference measurements on **one NVIDIA A100-SXM4-80GB**, at **864 x 480**,
using one denoising step, TAEHV encoding/decoding, SDPA, ring/fused DiT and CUDA
graphs, excluding startup and warmup:

| Scene | Prompt mode | Measured FPS | Measurement |
| --- | --- | --- | --- |
| Castle / Ancient Egypt | Fixed | ~38-39 | Model processing in an online session |
| Racing | Fixed | ~38.5 | Server-side online pipeline throughput |
| Racing | Dynamic | ~28.9 | Server-side online pipeline throughput |

Online pipeline throughput includes waiting for game frames; model-processing
throughput does not. Neither is browser presentation FPS or action-to-photon
latency. Racing figures come from an earlier checkpoint with the same model
architecture and inference configuration and serve as a performance reference.

## Layout

```text
matrix_game/inference/          # Shared offline/online model core
matrix_game/server/             # Game IPC, sessions and bounded media transport
matrix_game/web/                # Plain HTML/JS client, HUD and audio
matrix_game/configs/            # Scene names and fixed training-aligned prompts
game/                          # Standalone shooter source and shared dependencies
game/racing/                   # Standalone racer and its native IPC server
scripts/                       # Launch, resource and release tooling
tests/                         # CPU, protocol and browser regressions
releases/                      # Resource manifests, not large binary assets
```

See [architecture](docs/architecture.md) for the streaming contract and
[testing](docs/testing.md) for validation commands. Only one active game session
and one scene model per server process are supported.

## Security and License

This is a trusted single-user demo, not an authenticated public cloud-gaming
service. HTTP, RTSP and WebRTC bind to loopback by default. Do not expose them
directly to the Internet. See [SECURITY.md](SECURITY.md).

Project code is Apache-2.0; vendored components retain their original licenses.
Model and asset distributions carry their own accompanying license statements.
See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
