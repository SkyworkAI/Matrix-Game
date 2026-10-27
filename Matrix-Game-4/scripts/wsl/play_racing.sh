#!/usr/bin/env bash
# Install the native dependencies (once), build City Racing (once) and start it in WSLg, recording a take for `matrix-game render`.
# Run through scripts/wsl/play_racing.bat, or directly inside WSL (Ubuntu 22.04/24.04 with WSLg):
#   bash scripts/wsl/play_racing.sh [game arguments]
# With no arguments it starts the downtown time trial with the HUD and a 48 FPS offline recording into captures/. Any arguments you give
# replace those defaults; run `game/racing/bin/CitySampleBBoxGame --help` for the list.
# Keys: W/S accelerate/reverse, A/D steer, Space handbrake, F9 starts/stops recording, R recovers, Esc pauses.
# The first run asks for your sudo password once, to install the SDL2/OpenGL/FFmpeg development packages.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# the Windows checkout has CRLF line endings in its shell scripts, which bash rejects
for script in scripts/*.sh scripts/wsl/*.sh; do
  sed -i 's/\r$//' "${script}"
done

packages=(build-essential cmake pkg-config libsdl2-dev libglew-dev libgl-dev libegl-dev libavcodec-dev libavformat-dev libavutil-dev libswscale-dev zlib1g-dev ffmpeg)
missing=()
for package in "${packages[@]}"; do
  dpkg -s "${package}" >/dev/null 2>&1 || missing+=("${package}")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Installing: ${missing[*]}"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${missing[@]}"
fi

if [[ ! -d resources/racing/assets ]]; then
  echo "ERROR: resources/racing/assets is missing. Copy it from the DGX: scp -r kschmid@10.74.11.118:/localhome/kschmid/Matrix-Game/Matrix-Game-4/resources/racing resources/" >&2
  exit 1
fi

game="game/racing/bin/CitySampleBBoxGame"
if [[ ! -x "${game}" ]]; then
  echo "Building City Racing (first run only)..."
  BUILD_JOBS="$(nproc)" bash scripts/build_racing.sh -DRACING_ASSET_ROOT="${root}/resources/racing"
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "ERROR: no display. Run this from WSL with WSLg (Windows 11 or a recent Windows 10 with WSL 2)." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- --race --route downtown --hud --fps 48 --physics-hz 480 --offline-record captures
fi
mkdir -p captures
echo "Starting: ${game} --data-dir resources/racing $*"
"${game}" --data-dir resources/racing "$@"
echo "Takes so far:"
ls -d captures/take-* 2>/dev/null || echo "  none (press F9 in the game to start and stop a recording)"
