# Installation

Run the commands below from `Matrix-Game-4/` in the repository.

Use Python 3.12 for inference. The release dependency versions were taken from
the running service, not the old training repository's requirements file.
Install PyTorch with a CUDA build compatible with your NVIDIA driver, then the
extras shown in the [v4 README](../README.md). CUDA 13.2 / torch 2.12.0 is the source deployment
baseline; other combinations need numerical and throughput checks.

## Ubuntu Native Dependencies

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends \
  build-essential cmake pkg-config libsdl2-dev libglew-dev \
  libgl-dev libegl-dev libavcodec-dev libavformat-dev libavutil-dev ffmpeg
bash scripts/build_game.sh
```

Headless game capture needs **working hardware EGL/OpenGL**, not just CUDA.
For NVIDIA containers expose driver graphics libraries with the appropriate
NVIDIA container-runtime graphics capability. A software rasterizer may be
correct but far too slow. No proprietary driver libraries are included here.
Do not reuse an arbitrary host's copied driver files.

The browser server rejects known software renderers such as `llvmpipe` by
default. `--allow-software-rendering` is only for functional testing, not FPS
measurement. The actual OpenGL renderer is included in session statistics.

## macOS Standalone Game

```bash
brew install cmake pkg-config sdl2 ffmpeg
bash scripts/build_game.sh
```

The SDL/OpenGL game and offline recorder are supported separately from CUDA
inference. The neural model does not have a tested MPS path.

## Browser Media Sidecar

GStreamer runs in a separate Python process so its GI dependencies do not need
to be installed into the model environment:

```bash
sudo apt-get install --no-install-recommends \
  python3-gi python3-numpy gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-rtsp
```

`scripts/serve.sh` uses `/usr/bin/python3` for this sidecar by default. Override
`STUDIO_WEBRTC_PYTHON` with a GI-enabled interpreter when needed. It must import
NumPy; it does not need torch. GStreamer must provide `appsrc`, `videoconvert`,
`x264enc`, `h264parse` and `rtspclientsink`.

The sidecar clears inherited Python search-path/home overrides and disables
user-site imports to avoid mixing the model interpreter's binary extensions
with the GI interpreter. Install GI/NumPy into the chosen sidecar interpreter
or its system site, not only into a user-site directory.

Install MediaMTX from its [official releases](https://github.com/bluenviron/mediamtx/releases).
Pin the version used in your deployment and verify its upstream release hashes.
Run it with the supplied loopback-only configuration. No binary is downloaded
or executed implicitly by this project.

The model and game share a machine but may use separate GPUs. The model respects
`CUDA_VISIBLE_DEVICES`; this does not select the OpenGL device. Do not interpret
a two-GPU host as two-GPU DiT parallelism: this release's model runtime is single-GPU.
