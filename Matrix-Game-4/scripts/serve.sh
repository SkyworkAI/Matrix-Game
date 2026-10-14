#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export STUDIO_WEBRTC_PYTHON="${STUDIO_WEBRTC_PYTHON:-/usr/bin/python3}"
export STUDIO_WEBRTC_RTSP_PORT="${STUDIO_WEBRTC_RTSP_PORT:-8555}"
export STUDIO_WEBRTC_FRAME_PACING=clock
export STUDIO_MEDIA_MAXRATE_KBPS="${STUDIO_MEDIA_MAXRATE_KBPS:-2500}"
cd "$ROOT"
exec "${PYTHON:-python}" -m matrix_game.cli serve "$@"
