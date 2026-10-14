#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$ROOT/game/racing" -B "$ROOT/game/racing/build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$ROOT/game/racing/build" --parallel "${BUILD_JOBS:-4}"
ctest --test-dir "$ROOT/game/racing/build" --output-on-failure
