# Security Boundary

The browser demo is for one trusted user on a controlled host. It is not a
multi-tenant service. Keep HTTP, WHEP, ICE/TCP and RTSP on loopback, reached through
SSH. Same-origin/Host checks and packet validation are defense in depth, not user
authentication. Other users/processes on the host remain inside the trust boundary.

- Do not expose the ports publicly without authentication, TLS, authorization,
  resource quotas and a separately reviewed deployment.
- Run as an unprivileged user. The game/model should not need root.
- Load only expected model and game assets. Prefer the release safetensors;
  legacy PyTorch files use `weights_only=True` and must not be retried with
  unsafe pickle loading on failure.
- Resource downloads use an explicit manifest, immutable revision, SHA256 and
  byte-length checks. No download script executes remote code or extracts archives.
- Game control IPC is bounded JSON/binary, not pickle. Only display frames can
  be dropped when transport falls behind.
- Keep recordings, logs, credentials, environment files and internal paths out
  of Git and model cards. Local recordings contain player input/camera traces.
- Install MediaMTX/GStreamer/FFmpeg and GPU drivers from trusted distributors.
  Their security updates and deployment configuration remain the operator's duty.

Report security concerns through the publishing organization's private reporting
channel. Do not post tokens, private recordings or exploitable deployment details
in public issues.
