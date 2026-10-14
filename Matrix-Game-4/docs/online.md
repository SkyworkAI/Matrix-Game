# Online Demo

Run from `Matrix-Game-4/` in the repository. Build the game, obtain game assets and model
resources, and follow the media setup in [installation](installation.md).

```bash
mediamtx configs/mediamtx.yml
```

In a second terminal with the model environment active:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh --scene ancient-egypt --fps 40
```

The default preset matches the latest fast source deployment: one denoising step,
TAEHV encode/decode, ring/fused DiT, CUDA graphs and early RGB publication.
Change to `--scene castle` for Castle. There is no live checkpoint switching.
For `--scene racing`, build the separate racing executable and choose fixed or
dynamic captions as described in [City Racing](racing.md). Its first-frame sink
policy differs from the shooters. All scenes use single-GPU inference only.

For a no-model transport check, add `--reference-only`. A software OpenGL fallback
is rejected unless `--allow-software-rendering` is explicitly set for testing.
For two-step Wan rendering,
use `--preset quality`; reduce `--fps` to a sustainable measured capture rate.
Do not treat a target FPS setting as a model speed-up.

## Remote Browser Over SSH

Run this on the client, substituting your own server identity and port:

```bash
ssh -N -p SSH_PORT \
  -L 8025:127.0.0.1:8025 \
  -L 8890:127.0.0.1:8890 \
  -L 8190:127.0.0.1:8190 USER@HOST
```

Open `http://127.0.0.1:8025/`, press Play, then click the video for pointer lock.
The client supports a low-poly/render monitor, frame-aligned HUD, sound and
pointer/keyboard controls. Audio requires a browser user gesture.

Ports: HTTP 8025, WHEP 8890, ICE/TCP 8190, internal RTSP 8555. Forward the first
three only. When changing ports, update both server options and MediaMTX config;
the WHEP URL deliberately targets the client's loopback tunnel.

## Latency and Diagnostics

The display path is bounded but cannot eliminate physical network latency or
browser buffering. SSH/TCP may exhibit head-of-line blocking on lossy networks.
Measure generated FPS, encoded FPS and presented FPS separately; action-to-photo
latency also includes reference collection, inference, queueing and display.

`--profile-chunks` writes bounded per-chunk timings to `runtime/` on session close.
The browser's `window.castleDebug()` provides frame matching, presentation and
audio counters. These diagnostics must not be included in source releases.

Only one active session is allowed. Stop before starting another; disconnects
are cleaned up after the watchdog timeout. Startup model/graph warmup may take
time. Do not launch multiple model services on the same GPU accidentally.

This deployment assumes one trusted user. It intentionally has no public
registration, shared-session access, TLS termination or Internet-facing auth.
