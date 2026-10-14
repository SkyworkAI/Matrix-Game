#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$ROOT/game" -B "$ROOT/game/build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$ROOT/game/build" --parallel "${BUILD_JOBS:-4}"
ctest --test-dir "$ROOT/game/build" --output-on-failure
