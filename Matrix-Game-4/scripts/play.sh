#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MATRIX_GAME_DATA_DIR="${MATRIX_GAME_DATA_DIR:-$ROOT/resources/game}"
exec "$ROOT/game/bin/MedievalCastleBBoxMap" --offline-demo "$@"
