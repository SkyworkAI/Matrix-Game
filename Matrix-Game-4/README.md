# Matrix-Game 4.0

Streaming neural rendering for executable low-poly games.

[Project website](https://matrix-game-v4-0.github.io/page/)
| [Technical report](https://arxiv.org/abs/2610.27999)
| [Models and game assets](https://huggingface.co/Skywork/Matrix-Game-4.0)

The game engine owns gameplay, physics and input. A causal video model renders
the clean low-poly RGB stream; HUD and sound remain frame-aligned presentation
layers outside the model. This release includes **Medieval Castle**, **Ancient
Egypt**, and **City Racing**, with offline rendering and a single-user WebRTC
browser demo. Racing includes time trials and pursuit, plus fixed/dynamic prompts.

**2026-10-16:** Optimized inference for [RTX 5080](#rtx-5080) and
[RTX 5070 Ti](#rtx-5070-ti) is now available. With fixed prompts and SageAttention,
model throughput reaches up to **67 FPS on RTX 5080** and **58 FPS on RTX 5070 Ti**
at 864 x 480 (steady state, excluding video I/O). Use `--preset rtx5080-sage` or
`--preset rtx5070ti-sage`, respectively, with `--prompt-mode fixed`.

**2026-10-15:** Optimized inference for NVIDIA GeForce RTX 5090 is now available.
Use `--preset rtx5090`, or `--preset rtx5090-sage` with the optional SageAttention
backend. See [RTX 5090](#rtx-5090) for setup and launch commands.

## Requirements

- Model inference: Linux, NVIDIA GPU, Python 3.12. The A100 80 GB deployment uses
  PyTorch 2.12.0/CUDA 13.2. Dedicated RTX 5090, RTX 5080 and RTX 5070 Ti runtimes
  are also included; use the environment below and select the matching preset
  explicitly. RTX 5080 and RTX 5070 Ti support fixed prompts only.
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
driver. A separate FlashAttention installation is optional. The `rtx5090`,
`rtx5080` and `rtx5070ti` presets do not require SageAttention; their `-sage`
variants require the additional [SageAttention build](#optional-sageattention).
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
  --revision 2c53100457da7abcf0e966f75637c644d3423a63 \
  --scene ancient-egypt --resources resources

matrix-game download --manifest releases/models.json \
  --repo-id Skywork/Matrix-Game-4.0 --repo-type model \
  --revision 2c53100457da7abcf0e966f75637c644d3423a63 \
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
`--preset realtime` on A100, `--preset rtx5090` on RTX 5090, `--preset rtx5080`
on RTX 5080, or `--preset rtx5070ti` on RTX 5070 Ti. These presets support offline
and online rendering; hardware selection is not automatic.

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

On RTX GPUs, replace `--preset realtime` with the matching `rtx5090`, `rtx5080`
or `rtx5070ti` preset. See [RTX 5090](#rtx-5090), [RTX 5080](#rtx-5080) or
[RTX 5070 Ti](#rtx-5070-ti) for the optional SageAttention variants.

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
following frames. See [City Racing](docs/racing.md),
[architecture](docs/architecture.md) and [Future Sink](#future-sink-racing) for
prompt and temporal-position details.

## Future Sink (Racing)

Racing shows its sink image as frame zero and keeps it in the KV cache as the
visual reference for the whole session. With **future sink RoPE**, once the
21-latent window is full (output frame 85 onwards), the sink's keys are
re-rotated to 11 positions ahead of the current block, so the sink stays at a
constant distance from the frames being generated. With the previous
`absolute` policy the sink stays at position zero and its distance grows with
the session. In 4001-frame renders of two test drives with the default sink
image, `absolute` melted buildings and road surfaces after about 1800 to 2900
frames on both `realtime` and `quality`; `future` did not.

Future sink is the racing default, so no extra flag is needed. To choose the
policy explicitly, pass `--sink-rope-mode` to `matrix-game render` or
`scripts/serve.sh`. It works with every preset:

```bash
# Future sink (racing default)
matrix-game render --scene racing --input lowpoly.mp4 --resources resources \
  --preset realtime --prompt-mode fixed --sink-rope-mode future \
  --output outputs/racing-future

# Previous behavior, for comparison
matrix-game render --scene racing --input lowpoly.mp4 --resources resources \
  --preset realtime --prompt-mode fixed --sink-rope-mode absolute \
  --output outputs/racing-absolute
```

| `--sink-rope-mode` | Sink keys once the window is full | Default for |
| --- | --- | --- |
| `future` | 11 positions ahead of the current block | Racing |
| `fixed` | 21 positions behind the current block | Castle, Ancient Egypt |
| `absolute` | Position zero; the distance grows with the session | None |

Only the sink's keys move. Its values and the window frames keep their
positions, the weights are unchanged, and the policies differ only from output
frame 85 onwards. The offset is `FUTURE_SINK_OFFSET` in
`matrix_game/inference/sink_policy.py`, and `render.json` records the policy as
`sink_rope_mode`. A sink image is required: `--sink-image`, or
`models/<scene>/sink.png` by default.

## Inference with Reactor Runtime

The optional [Reactor integration](reactor/README.md) serves the original game
controls and streaming neural renderer through [Reactor](https://reactor.inc).
Its browser client presents the low-poly draft and generated view side by side
in the same video frames. It has been tested on one B200 with native-output
comparisons and live controls; see the integration guide for measured results
and the observed long-rollout visual limitations.

With Docker, NVIDIA Container Toolkit and the
[Reactor CLI](https://docs.reactor.inc/deploy/platform/installation) installed,
run these commands from `Matrix-Game-4/`. The service loads the three released
scenes for selection in the client:

```bash
cd reactor
reactor build
reactor run --gpus device=0 --port 8080
```

Wait for `http://localhost:8080/health` to report `"state":"available"`.
In a second terminal, from `Matrix-Game-4/`:

```bash
cd reactor/demo
npm ci
npm run dev -- --host 127.0.0.1
```

Open the printed frontend URL, normally `http://localhost:5173`. Click
**Connect**, select a scene card, seed and any racing options, then **Start game**.
Click the video to capture keyboard and mouse input; Esc releases them. The
on-screen buttons show live inputs and can also be held to control the game.
Use **Stop → choose scene → Start game** to switch scenes.
For SSH access, follow the guide's [HTTP and WebRTC instructions](reactor/README.md#remote-access-over-ssh).

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

## RTX 5090

Start with `--preset rtx5090` after the standard installation and resource
downloads above. It uses the same scene checkpoints and shared resources; no
separate quantized checkpoint is needed. DiT block linear weights are converted
to FP8 at load time, and their activations are quantized during inference.
Embeddings and the output head remain BF16.

Both RTX 5090 presets select TAEHV encoding/decoding, one-step denoising, fused
kernels and CUDA graphs. They differ in the self-attention backend:

| Preset | Self-attention | Additional setup |
| --- | --- | --- |
| `rtx5090` | PyTorch SDPA | None beyond the standard installation |
| `rtx5090-sage` | SageAttention for the full history window; SDPA while it fills | Patched SageAttention build below |

Render an existing low-poly video or start the browser server:

```bash
matrix-game render --scene ancient-egypt --input lowpoly.mp4 \
  --resources resources --preset rtx5090 --prompt-mode fixed \
  --output outputs/egypt-rtx5090

CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh \
  --scene ancient-egypt --resources resources --preset rtx5090 \
  --prompt-mode fixed --fps 40
```

The browser server uses the same native-game and MediaMTX setup described above.
The runtime warms up kernels and captures CUDA graphs before accepting a real
session, so startup takes longer than steady-state rendering. `--fps 40` still
controls the game capture cadence, not measured rendering throughput.

### Optional SageAttention

To use `--preset rtx5090-sage`, build SageAttention 2.2.0 with the supplied
CUDA-graph patch and CUDA 13.2. Run these commands from `Matrix-Game-4/` in the
same Python environment as the model:

```bash
python -m pip install "nvidia-cuda-nvcc==13.2.*" "nvidia-nvvm==13.2.*" "nvidia-cuda-crt==13.2.*" "nvidia-cuda-cccl==13.2.*" setuptools wheel ninja
export CUDA_HOME="$(python -c 'import nvidia, os; print(os.path.join(nvidia.__path__[0], "cu13"))')"
mkdir -p ../cuda-link && ln -sf "$CUDA_HOME/lib/libcudart.so.13" ../cuda-link/libcudart.so
git clone https://github.com/thu-ml/SageAttention.git ../SageAttention
git -C ../SageAttention checkout eb615cf6cf4d221338033340ee2de1c37fbdba4a
git -C ../SageAttention apply "$PWD/matrix_game/inference/rtx5090_sage/sageattention.patch"
LIBRARY_PATH="$PWD/../cuda-link" MAX_JOBS=4 EXT_PARALLEL=2 python -m pip install --no-build-isolation ../SageAttention
```

Then replace `--preset rtx5090` with `--preset rtx5090-sage` in either command
above. No SageAttention build is needed when using plain `rtx5090`.

## RTX 5080

Use `--preset rtx5080`, or `--preset rtx5080-sage` with the
[SageAttention build](#optional-sageattention). Fixed prompts only (`--prompt-mode fixed`):
with 16 GB, T5 encodes the prompt and is released before the DiT is loaded.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh \
  --scene ancient-egypt --resources resources --preset rtx5080-sage \
  --prompt-mode fixed --fps 40
```

| Preset | Model FPS | GPU memory |
| --- | --- | --- |
| `rtx5080` | ~42 | ~9.6 GB |
| `rtx5080-sage` | ~67 | ~12.8 GB |

Model throughput on one RTX 5080 at 864 x 480, steady state, excluding video I/O.

## RTX 5070 Ti

Use `--preset rtx5070ti`, or `--preset rtx5070ti-sage` with the
[SageAttention build](#optional-sageattention). Fixed prompts only, as on the RTX 5080.
The block linears use INT8 (W8A8, no calibration), which is faster than FP8 at this
model's GEMM shapes on this GPU; tuning is in `matrix_game/inference/rtx5070ti/config.py`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh \
  --scene ancient-egypt --resources resources --preset rtx5070ti-sage \
  --prompt-mode fixed --fps 40
```

| Preset | Model FPS | GPU memory |
| --- | --- | --- |
| `rtx5070ti` | ~41 | ~9.2 GB |
| `rtx5070ti-sage` | ~58 | ~10.9 GB |

Model throughput on one RTX 5070 Ti at 864 x 480, steady state, excluding video I/O.

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

## Citation

If you use Matrix-Game 4.0 in your research, please cite:

```bibtex
@misc{zhang2026matrixgame40harnessingvideo,
      title={Matrix-Game 4.0: Harnessing Video Models for Real-Time Playable Game Worlds},
      author={Jihai Zhang and Runjia Qian and Kaiqin Kong and Biao Jiang and Puyi Wang and Siru Chen and Tianle Li and Guanjie Chen and Junyi Wu and Chubin Chen and Yuanhang Li and Kaichen Huang and Haobo Zhang and Yangguang Li and Wei Yu and Guibin Chen and Yu Cheng and Yahui Zhou},
      year={2026},
      eprint={2610.27999},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.27999},
}
```

## Security and License

This is a trusted single-user demo, not an authenticated public cloud-gaming
service. HTTP, RTSP and WebRTC bind to loopback by default. Do not expose them
directly to the Internet. See [SECURITY.md](SECURITY.md).

Project code is Apache-2.0; vendored components retain their original licenses.
Model and asset distributions carry their own accompanying license statements.
See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
