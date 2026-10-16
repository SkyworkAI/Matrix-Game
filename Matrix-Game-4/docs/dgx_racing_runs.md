# City Racing on the DGX and WSL: setup, recording, rendering and measurements

Host: DGX GB300 (`kschmid@10.74.11.118`, aarch64). Repo there: `/localhome/kschmid/Matrix-Game/Matrix-Game-4`. Local repo: `C:\workspace\world\Matrix-Game\Matrix-Game-4`. Everything below was run on 2026-10-15 and 2026-10-16. Nothing here is validated by the model authors.

## Scripts added to this checkout

| Script | Purpose |
| --- | --- |
| `scripts/dgx/setup_dgx.sh` | uv venv with Python 3.12, torch 2.12.0 / torchvision 0.27.0 (`CUDA_TAG` default `cu132`), then the package with its extras; stops unless a bf16 matmul works on the GPU. |
| `scripts/dgx/download_scene.sh <racing\|castle\|ancient-egypt\|all>` | Downloads and verifies one scene's checkpoint and the game assets at the pinned revision. Runs `link_cached_base.py` first. |
| `scripts/dgx/link_cached_base.py` | Hard-links the T5 encoder and Wan2.2 VAE from the Hugging Face cache when their SHA256 matches the manifest. |
| `scripts/dgx/run_racing.sh` | One render: `--take DIR` or `--input VIDEO`, `--preset`, `--output DIR`; checks free GPU memory (30000 MiB). Extra render arguments go after `--`. |
| `scripts/dgx/make_test_input.py` | Synthetic low-poly stand-in clip (4k+1 frames). Exercises the pipeline only. |
| `scripts/dgx/upscale_video.py` | Real-ESRGAN 2x post-process (weights loaded through `spandrel`). Not used in the final renders. |
| `scripts/dgx/benchmark_stream.py` | Measures chunk time, stage times, FPS and streaming latency on a take. |
| `scripts/wsl/play_racing.bat` / `.sh` | Installs the native packages once (sudo), builds the racing game in WSL and starts it in WSLg, recording takes to `captures/`. |
| `scripts/wsl/render_take.bat` | Copies a take to the DGX, renders it, copies `rendered.mp4` and `render.json` back to `outputs\<name>\`. Picks a free output name (`<name>`, `<name>_2`, ...). `SINK_IMAGE` sets a reference image; `MATRIX_GAME_RENDER_SIZE` sets the size. |
| `scripts/ui/render_ui.py` / `.bat` | Local web UI (127.0.0.1 only) for offline renders: pick a take, preset, steps, size, prompt, seed and reference image; Render runs the same steps as `render_take.bat`; Measure latency runs `benchmark_stream.py` on the DGX for the chosen settings. Plays past renders two at a time against the take's input. Needs ssh, scp, ffmpeg. |
| `scripts/dgx/start_live.sh` | Starts/stops/checks MediaMTX and `matrix-game serve` on the DGX. Idempotent: leaves an already-running stack alone. `start\|stop\|status`, `--preset`, `--fps`, `--prompt-mode`. Only stops processes it itself started (tracked by pid files in `runtime/live/`). |
| `scripts/wsl/start_live.bat` | Runs `start_live.sh` over ssh, opens the SSH tunnel (ports 8025, 8890, 8190) if not already open, opens the browser. Same subcommands as the DGX script. |

## Local changes to shipped code

- `matrix_game/inference/resolution.py` (new), `stream.py`, `offline.py`, `vae_graph.py`: the render size comes from `MATRIX_GAME_RENDER_SIZE` (for example `1728x960`, sides multiples of 32). Default is 864x480. The `rtx5090` stream path still has 864x480 hardcoded. The checkpoint was trained at 864x480; other sizes are unvalidated.
- `game/racing/src/MainSDL.cpp` line 121 (DGX copy only): offline recording no longer rejects `--autodrive`, so the game's route-following bot can drive a recorded take headlessly.
- `matrix_game/web/index.html`, `racing.js` (live page, both the local checkout and the DGX copy under `.../matrix_game/web/`): English route names (`ROUTE_LABELS` in `racing.js`, mapped from `route_id`, since the shipped names are Chinese text from `race-routes.json`); an on-screen key display (W/A/S/D/Space/R, lit while held); a latency panel (lag now/median/p95/max in ms, shown and render FPS); per-frame latency recorded to a CSV that auto-downloads on Stop or via the "Save latency" button; one summary row per session (total latency and FPS) kept in `localStorage` and downloadable via "Session log". None of this touches the server (`matrix_game/server/*.py`); it hot-reloads by refreshing the page.

## Pipeline

Offline render (`matrix-game render`):

`game take (reference.mkv)` -> `VAE encode (1 frame, then 4 per chunk)` -> control latent -> `causal DiT` <- text (T5) -> latent -> `VAE decode` -> `rendered.mp4`. The reference (sink) image is VAE-encoded, decoded as output frame 0 with no denoising, and written into the DiT cache as clean memory, so every later frame is conditioned on it. The prompt reaches the DiT only by cross-attention.

- Game: `game/racing/src/MainSDL.cpp` (game and recorder), `ReferenceServerMain.cpp` (live server).
- Harness (writes the prompt from game state): `game/racing/src/CaptionStream.cpp`, `WorldCaptions.cpp`, `matrix_game/server/racing.py`, then `stream.set_prompt()`. Offline it is the fixed caption, `--prompt`, or `--prompt-mode dynamic --captions captions.jsonl` (one record per 4-frame chunk; takes contain no captions).
- Reference image: `resources/models/racing/sink.png` by default, or `--sink-image FILE`. It is not frame 0 of `reference.mkv`.
- Model: `CastleStream` in `matrix_game/inference/stream.py`; 5.0 B parameters, all BF16 (825 tensors, 10.0 GB); base "Wan2.2-Fun-5B-Control causal"; latent 48 x 30 x 54 at 864x480, one latent per chunk (4 frames, 1 for the first chunk); 21-latent cache including the sink slot; CFG 1; shift-5 timesteps. The released weights are not quantized. The `rtx5090` preset converts DiT linear layers to FP8 at load time.

Live play (`matrix-game serve`, running on the DGX since 2026-10-15):

game (headless, under the server) -> low-poly RGBA and metadata over a private socketpair (length-prefixed JSON plus raw pixels) -> preprocess (one chunk lookahead) -> model -> decoded frames (4-slot drop queue) -> GStreamer -> MediaMTX (RTSP/WHEP) -> browser at `http://127.0.0.1:8025/`, with keys and mouse sent back to the game. If the model falls behind the game slows down; only display frames are dropped. Needs MediaMTX, GStreamer with RTSP plugins, working headless OpenGL, and an SSH tunnel for ports 8025, 8890 and 8190 (`docs/online.md`).

### Live setup, done once on the DGX

1. `sudo apt-get install -y --no-install-recommends python3-gi python3-numpy gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-rtsp` (log: `outputs/apt_live_deps.log`). `sudo` is passwordless on this host.
2. MediaMTX is not in apt. Downloaded `mediamtx_v1.21.2_linux_arm64.tar.gz` from the `bluenviron/mediamtx` GitHub releases, checked its SHA256 against the release's `checksums.sha256`, and installed the binary at `~/bin/mediamtx` (no `sudo` needed). Use `linux_amd64` on an x86_64 host.
3. Start both with `bash scripts/dgx/start_live.sh` (or `scripts\wsl\start_live.bat` from Windows, which also opens the tunnel and the browser). It checks for at least `MIN_FREE_MIB` (default 30000) of free GPU memory before starting the model.

### Live routes and modes

| Mode | What it is | Route | Ends when |
| --- | --- | --- | --- |
| Time trial | Numbered checkpoints (65 on Downtown), elapsed time, penalty seconds | Downtown, Avenue or Skyway | last checkpoint passed |
| Pursuit | Chase and hold capture distance for 5 s | Avenue only (route selector is disabled) | capture held 5 s, or defeated |
| Free drive | No objective | any | Stop pressed |

R recovers in a time trial (+5 s), is disabled in pursuit. P pauses, Enter resumes. Route ids map to English names in the HUD (see web changes above); the underlying `race-routes.json` names/descriptions stay Chinese and are not displayed.

### Event (per-chunk) prompts

Dynamic mode gives the prompt a per-4-frame-chunk update instead of one fixed caption for the whole clip, which is how a weather or car-colour look can be held through a long take (our fixed-prompt snow renders faded back to the default look after about a quarter of the clip - see Known issues).

- Live: `--prompt-mode dynamic --prompt-interval-chunks N` (also exposed by `start_live.sh --prompt-mode dynamic`). The game itself writes the caption each chunk from the objects actually visible (`game/racing/src/CaptionStream.cpp`, `WorldCaptions.cpp`), not from custom events; this costs throughput (README A100 reference: ~38.5 FPS fixed vs ~28.9 FPS dynamic for racing).
- Offline: `--prompt-mode dynamic --captions captions.jsonl` on `run_racing.sh`/`benchmark_stream.py`. One JSON object per line, `frame_start`/`frame_end` (inclusive) covering frame 0 then consecutive groups of 4, each `prompt` non-empty and under 8192 characters. To hold a look, repeat the base description (weather, car, setting) in every line and vary only the event:
  ```
  {"frame_start":0,"frame_end":0,"prompt":"Heavy snow in a city, snow-covered road. Center: blue sedan seen from behind, driving straight."}
  {"frame_start":1,"frame_end":4,"prompt":"Heavy snow in a city, snow-covered road. Center: blue sedan seen from behind, accelerating."}
  ```
  Not yet tried on a real take; the plan is to generate it from a take's `frames.jsonl` (speed, steering, collisions, checkpoint progress).

### Render UI and latency logging

`scripts/ui/render_ui.py` (`scripts\ui\render_ui.bat`) serves a page at `http://127.0.0.1:8765/` for offline renders, with a "Measure latency" button that runs `benchmark_stream.py` on the DGX for the selected take/preset/steps/size and shows chunk time, per-stage time, FPS and streaming latency (see Measurements table below for the same numbers run by hand). Tested via its HTTP API and one end-to-end `realtime` render (399 frames, finished in the UI); the page itself (buttons, two-up players) has not been opened in a browser.

The live page (`http://127.0.0.1:8025/`) separately logs latency per browser session: a panel shows live lag (now/median/p95/max, ms) and FPS; every displayed frame's lag is recorded and downloads as a CSV on Stop or via "Save latency"; one summary row per session (duration, video source, mode, route, sample count, mean/median/p95/max lag, mean/min shown FPS, mean render FPS) is kept in the browser's `localStorage` and downloadable via "Session log". "Lag" here is input-sent-to-frame-displayed, so it includes the SSH tunnel, the game, the model and video encode/decode - not run yet with real numbers recorded.

## Measurements (DGX GB300, racing, take-001, 24 fps source)

Per-chunk time is the host wall time of one `push` after the GPU finished. FPS (model only) is 4 frames divided by the mean chunk time. Latency is the 167 ms needed to fill a 4-frame chunk at 24 fps plus the processing time; it excludes network, encode and display.

864x480 (100 chunks, first 30 excluded):

| Preset | VAE | DiT runtime | Steps | Chunk p50 | Chunk p95 | FPS (model) | FPS in real renders | First chunk | Latency | Real time at 24 fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `realtime` | TAEHV | ring-fused | 1 | 30.8 ms | 31.0 ms | 129.9 | about 44 | 114 ms | about 197 ms | Yes |
| `quality` | Wan | native | 4 | 356.6 ms | 359.7 ms | 11.2 | about 10.5 | 407 ms | about 523 ms | No |
| `quality` | Wan | native | 8 | 556.9 ms | 560.1 ms | 7.2 | about 6.9 | 404 ms | about 724 ms | No |

1728x960 (60 chunks, first 15 excluded):

| Preset | Steps | Chunk p50 | Chunk p95 | FPS (model) | First chunk | Latency | Real time at 24 fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `realtime` | 1 | 101.0 ms | 148.2 ms | 37.6 | 113 ms | about 268 ms | Yes |
| `quality` | 4 | 544.0 ms | 582.0 ms | 7.3 | 424 ms | about 711 ms | No |
| `quality` | 8 | 773.6 ms | 811.8 ms | 5.1 | 429 ms | about 940 ms | No |

Stage medians per chunk (ms), 864x480: realtime encode 1.7, denoise 13.3, clean KV 12.0, decode 1.5; quality 4 steps encode 18.1, denoise 226.6, clean KV 54.5, decode 54.9; quality 8 steps denoise 428.8. At 1728x960 the VAE stages grow about 3.7x, but 4-step denoising did not get slower (215 ms); the reason was not investigated. Raw reports: `outputs/bench_*.json`, `outputs/bench1728_*.json`.

## Reference images

One image per render or live session, no more. Checked every file that touches `sink_image` (`matrix_game/scenes.py`, `matrix_game/inference/loader.py`, `matrix_game/inference/stream.py`, `matrix_game/inference/rtx5090/stream.py`): `--sink-image` is `type=Path`, loaded with one `Image.open()` call and hashed with one `read_bytes()`; `CastleStream`/`FastStream` take a single `sink_latent`, no list or registry. No code path accepts a folder, a sequence of images, or a video as the reference. `--no-sink-image` falls back to a `"generated_first_frame"` sink source (the model generates its own opening frame) instead of an encoded picture - not tried. The reference image cannot be swapped mid-clip; two looks in one output would mean rendering twice and splicing.

A richer design - several reference frames plus pose-indexed spatial memory, shown in a diagram the user had (source unclear, matches the project page preview text rather than this release) - is not implemented here. The closest piece that exists in the code is `memory_mode="kvbank"` in `matrix_game/inference/rollout.py` (`CausalV2VRollout`, alternative to `"none"`), but `CastleStream` hardcodes `memory_mode="none"` and nothing in the repo (CLI, scene config, server) ever sets `"kvbank"`, so it is dead code on every path we can run. No pose-indexed logic exists anywhere in `matrix_game/inference/`. A `RegisterTokens`/`adding_cls_branch` class in `wrappers.py` looks related to the diagram's "Register" box by name only - it is explicitly commented as hard-coded training-time GAN-classifier code for a different model (WAN2.1-T2V-1.3B) and is never called from the render or serve paths.

- Racing default: `resources/models/racing/sink.png` (grey sedan, cloudy downtown street, 1280x720).
- Sunny taxi (user made): `C:\Users\kschmid\Downloads\Gemini_Generated_Image_e3sd2ce3sd2ce3sd.jpg`.
- Snowy blue sedan (user made): `outputs/e2ea3bb9-c2fe-40b2-8d36-91c4b325247c.jpg`.
- Castle and Egypt (not racing): `captures/other_sinks/fps-castle/sink.png` and `fps-egypt/sink.png` on the DGX.

## Renders (quality preset, Wan VAE, seed 0, 864x480 unless noted)

| Output folder | Take | Steps | Reference | Result |
| --- | --- | --- | --- | --- |
| `racing_unrealcity1` | gray clip | 1 (realtime) | default | blurry |
| `racing_unrealcity_hq`, `racing_unrealcity_nyc_taxi` | gray clip | 4 | default | sharper; follows the clip loosely |
| `racing_take001_hq_2` | take-001 | 4 | default | good, 399 frames |
| `racing_take001_taxi_sunny` | take-001 | 4 | sunny taxi | yellow taxi and sunlight hold; a blue roof beacon appears after frame 0 |
| `racing_take001_snow`, `racing_take001_snow2` | take-001 | 4 | snowy sedan | snow holds for about a quarter of the clip, then fades to dry streets |
| `racing_take002_8step`, `racing_take002_yellow_sunny` | take-002 (autodrive) | 8 | default | drifts mid-clip (wrong scene, grey car) |
| `racing_take002_1728` | take-002 | 4 | default | native 1728x960; frame 300 sharp, then drifts to a car interior around frame 700 |

## The `vae_decoder` file on Hugging Face

`Skywork/Matrix-Game-4.0/driving_game/vae_decoder/decoder.pth` (22.9 MB) is not referenced by any manifest or by the Python code (only `base/taew2_2.pth` is loaded). Downloaded and compared tensor-by-tensor with `taew2_2.pth`: same 128 tensor names/shapes, all float16 (not quantized), the 64 `encoder.*` tensors are bit-identical, the 64 `decoder.*` tensors all differ by small amounts (first differing tensor: max abs diff 0.042). Reading: a racing-scene fine-tune of the TAEHV decoder half, shipped but unused by the released inference code. Not tried in a render (`--taehv-path` would point at it).

## WSL (local, same RTX 5090 as the game build)

Repo copied to `/home/kschmid/Matrix-Game-4` (rsync, excluding `outputs/`, `captures/`, `resources/`, `.venv/`, game build/bin). `scripts/dgx/setup_dgx.sh` run there too (same script, aarch64-agnostic): Python 3.12 venv, torch 2.12.0/cu132, `matrix-game` command installed and verified against the GPU. Weights not downloaded yet (would need ~25 GB and the GPU free). The 32 GB card is shared by other processes on this machine; check `nvidia-smi` before any run here. Whether the `quality` (bf16) preset fits in 32 GB at 864x480 is unconfirmed; `rtx5090`/`rtx5090-sage` (FP8 DiT linears, same checkpoints) are the README's recommended presets for this GPU.

## Known issues

- Long or fast drives drift away from the input (scene layout, car, weather). A fixed prompt cannot hold a weather look; per-chunk captions (above) are the untested next step.
- The seed-1 NYC-taxi rerun of the gray clip failed (exit code 4) and the cause was not found.
- The DGX checkout was replaced between runs on 2026-10-15 (everything outside git, including `.venv`, `resources/`, `outputs/` and `captures/`, was gone; a `~/Matrix-Game/build_all.sh` not written by us appeared, dated after the loss). Rebuilt with the scripts above. Keep copies of takes and renders on Windows; the DGX is not a safe store.
- A render needs a fresh output folder; `render_take.bat` and the render UI pick one automatically (`<name>`, `<name>_2`, ...).
- The live stack (MediaMTX, `matrix-game serve`) was started by hand before `start_live.sh` existed, so it has no pid files; `start_live.sh stop` cannot stop it. The next stop/restart should go through the script so pid tracking takes over.

## Data and training

The repository ships no dataset, no sample recording and no training code (`docs/installation.md` mentions a separate, old training repository). The project page (https://matrix-game-v4-0.github.io/page/) is a preview; the paper is "coming soon" and does not disclose data or training stages. `inference.json` names the architecture as "Wan2.2-Fun-5B-Control causal"; all shipped checkpoints (racing, castle, ancient-egypt, and the `driving_game/vae_decoder` extra above) are full-precision, not quantized.
