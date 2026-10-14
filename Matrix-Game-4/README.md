# Matrix-Game 4.0

Streaming neural rendering for executable low-poly games.

[Project website](https://matrix-game-v4-0.github.io/page/)
| [Models and game assets](https://huggingface.co/Skywork/Matrix-Game-4.0)

The game engine owns gameplay, physics and input. A causal video model renders
the clean low-poly RGB stream; HUD and sound remain frame-aligned presentation
layers outside the model. This release includes **Medieval Castle**, **Ancient
Egypt**, and **City Racing**, with offline rendering and a single-user WebRTC
browser demo. Racing includes time trials and pursuit, plus fixed/dynamic prompts.

> The project website describes the broader project; this package does not
> implement a VLM agent or camera-retrieved world-memory bank.

## Requirements

- Model inference: Linux, NVIDIA GPU, Python 3.12. The source deployment used
  A100 80 GB and PyTorch 2.12.0/CUDA 13.2. Consumer-GPU fit is not verified here.
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
  base/                         # Shared umT5, tokenizer, Wan VAE, TAEHV
  models/castle/                 # model.safetensors, sink.png
  models/ancient-egypt/          # model.safetensors, sink.png
  models/racing/                # model.safetensors, sink.png
  game/assets/                  # Shooter assets and per-scene maps
  racing/assets/                # City map, captions, routes and vehicle audio
```

Model weights, the fast codec, sink images and game assets are hosted on
[Hugging Face](https://huggingface.co/Skywork/Matrix-Game-4.0). Shared T5/tokenizer
and Wan VAE files are downloaded automatically from a pinned official base-model
revision rather than duplicated in this repository.
Download instructions and the pinned revision are in
[releases/README.md](releases/README.md). The resource downloader verifies file
lengths and SHA256, requires an immutable Hub revision and downloads only the
selected scene plus shared assets. An existing resource directory can be checked:

```bash
matrix-game verify --manifest releases/game-assets.json --resources resources
```

## Offline Rendering

```bash
matrix-game render --scene ancient-egypt --input lowpoly.mp4 \
  --resources resources --preset quality --output outputs/egypt
```

The `quality` preset uses Wan VAE encoding/decoding, two denoising steps
(`1000,500`, including startup), and native DiT. Output is `rendered.mp4` plus a
configuration report. FPS and frame count are preserved; only the final incomplete
four-frame block is padded internally, then trimmed from the output.

To render a local game recording with the scene read from its manifest:

```bash
matrix-game render --take /path/to/take-1 --resources resources \
  --preset quality --output outputs/take-1
python game/tools/finalize-demo.py /path/to/take-1 \
  --rendered outputs/take-1/rendered.mp4
```

See [offline recording](docs/offline.md) for clean RGB, HUD, audio and input traces.

## Play and Record Locally

```bash
bash scripts/build_game.sh
bash scripts/play.sh --scene ancient-egypt
```

`play.sh` starts the existing **48 FPS / 2x simulation-time** demo mode without
neural rendering or simulated network delay. F9 toggles recording; Escape opens
the menu. Use `--scene castle` for the other map. The game can run without Python
model dependencies. `MATRIX_GAME_DATA_DIR` overrides `resources/game`.

For racing, use `bash scripts/build_racing.sh`, then follow
[City Racing](docs/racing.md) for local play/recording and the browser modes.

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

The `realtime` preset uses TAEHV encoding/decoding, one denoising step,
ring/fused DiT, CUDA graph replay and early RGB publication. Shooter startup
also uses one step; racing displays its supplied sink as frame zero without
denoising, then uses one step per block. `--fps 40`
is a **target capture cadence, not a throughput or latency guarantee**. The first
chunk contains one RGB frame; subsequent chunks contain four. Ring/fused math
can differ numerically from native arithmetic; it is not bitwise-equivalent.

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
