# Runtime Contract

`native game -> clean RGB + frame metadata -> causal renderer -> bounded display
publisher -> WebRTC -> presented-frame HUD/audio`

The shooter uses 120 Hz physics. Online capture advances an integer number
of physics steps per frame, so supported target rates divide 120 (for example
20, 24, 30, 40 or 60 FPS). Shooter local recording is a separate 48 FPS / 2x mode.
Racing selects a physics rate divisible by the requested capture cadence and
supports 1-60 FPS online. Local racing recordings use their configured cadence.

## Shooter Model

- 480x864 output; low-poly RGBA capture is 1280x720, converted to RGB, bilinearly
  resized and center-cropped using the training preprocessing implementation.
- Wan2.2 5B latent space: 48 channels, 30x54 spatial resolution, temporal factor 4.
- One latent per chunk: first RGB chunk is one frame; later chunks are four frames.
- No supplied high-quality gameplay first frame. The independently encoded sink
  image is an identity prefix in KV only; it is not rendered or used as a control
  video frame, and does not seed the streaming VAE feature cache.
- Fixed sink RoPE, a bounded 21-latent cache including the sink, fixed scene prompt,
  CFG 1, shift-5 timestep mapping. The clean prediction is written back with zero
  noise and `drop_control=True` after each chunk.
- No automatic cache reset, SDEdit initialization, retrieved memory or dynamic
  prompt generation in the released gameplay configuration.
- A session has a finite RoPE budget (8192 generated latent chunks by default).
  Start a new session after reaching it; do not silently wrap positional indices.

TAEHV is an approximate codec, not numerically identical to Wan VAE. `quality`
uses Wan codecs and two steps; `realtime` uses TAEHV and one step. Fast runtime
modes alter floating-point reduction order. Keep `native` for reference checks.
## Racing Model

Racing shares the backbone, codec wrappers, ring/fused kernels and CUDA graphs,
but uses **future sink RoPE**: once the 21-latent window is full, the sink's keys
are re-rotated to 11 positions ahead of the current block
(`matrix_game/inference/sink_policy.py`). Its supplied sink is encoded by the selected
codec, decoded as frame zero, and committed to clean KV without denoising.
It seeds decoder history, not the independent low-poly encoder history. The
realtime preset uses TAEHV for the sink, reference encoding and output decoding.
Native capture is 1920x1080, downsampled to 864x480 by the original game pipeline.

Fixed mode frees T5 after startup. Dynamic mode keeps T5 resident, uses exact
chunk-aligned engine captions, and refreshes cross-attention KV only when the
text changes. A bounded eight-caption cache avoids redundant encodes. Graph
replay retains native 512-token padding and stable cross-KV addresses.

The same `CastleStream` core and model loader serve both online and offline paths;
the class's historical name does not imply a castle-specific architecture.

## Synchronization and Backpressure

Game IPC is length-prefixed JSON plus raw pixels over a private socketpair.
It accepts only bounded packets and consecutive frame indices. Preprocessing
uses one chunk of lookahead; it neither skips model conditioning frames nor
changes their simulation-time cadence. When production is blocked, the game
slows relative to wall time instead of running a catch-up burst.

Only **decoded display frames** can be dropped. Their queue has four slots;
the media subprocess and browser never block model inference on network delivery.
Frame metadata is associated with the actual RTP timestamp. The browser matches
HUD and audio when a video frame is presented, not when a message arrives.

Shooter mouse deltas are sent at approximately 120 Hz while active, consumed at the
physics rate and bounded to 50 ms of pending rotation. Camera limit: 90 degrees
per simulation second. Keyboard/button edges are sent immediately. Loss of
focus resets controls. Shooter sounds are synthesized; racing uses attributed
vehicle samples with telemetry-driven pitch/gain. Both follow presented frames.

## Ownership

The inference package does not import the game, HTTP or GStreamer. The native
game does not import Python. The server owns one session and connects the two.
The media sidecar imports only standard Python, NumPy and GI. Offline rendering
does not require aiohttp, GStreamer or a running game.
