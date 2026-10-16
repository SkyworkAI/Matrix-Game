#!/usr/bin/env bash
# Render the Matrix-Game 4.0 City Racing scene on the DGX, offline, with the README's fast "realtime" preset (TAEHV codec, one denoising
# step, accelerated DiT) and the fixed scene prompt. Needs scripts/dgx/setup_dgx.sh and `scripts/dgx/download_scene.sh racing` first.
#
# The model renders a clean low-poly RGB video (or a recorded game take). The repository ships no sample recording and the native game
# cannot be built on this host, so without --input or --take this script writes a synthetic stand-in clip (make_test_input.py). That clip only
# exercises the pipeline; use a real recording for real results.
#
# Usage:   bash scripts/dgx/run_racing.sh [--input lowpoly.mp4 | --take captures/take-001] [--seconds N] [--preset realtime|quality]
#                                         [--output DIR] [-- further matrix-game render arguments]
#   --seconds N   length of the synthetic clip when no input is given (default 6)
#   --preset P    realtime (default) or quality (Wan VAE, two denoising steps, native DiT: slower)
#   --output DIR  default outputs/racing_<timestamp>; must not exist yet (the render creates it). The log is <DIR>.render.log
# The GPU is checked first: the run refuses to start with less than MIN_FREE_MIB (default 30000) of free GPU memory.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

input=""
take=""
seconds=6
preset="realtime"
output=""
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) input="${2:?--input needs a file}"; shift 2 ;;
    --take) take="${2:?--take needs a directory}"; shift 2 ;;
    --seconds) seconds="${2:?--seconds needs a number}"; shift 2 ;;
    --preset) preset="${2:?--preset needs a name}"; shift 2 ;;
    --output) output="${2:?--output needs a directory}"; shift 2 ;;
    --) shift; extra=("$@"); break ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

python_bin=".venv/bin/python"
tool=".venv/bin/matrix-game"
[[ -x "${tool}" ]] || { echo "ERROR: ${tool} not found. Run scripts/dgx/setup_dgx.sh first." >&2; exit 1; }
for required in resources/models/racing/model.safetensors resources/base/models_t5_umt5-xxl-enc-bf16.pth resources/base/Wan2.2_VAE.pth resources/racing/assets; do
  [[ -e "${required}" ]] || { echo "ERROR: ${required} is missing. Run: bash scripts/dgx/download_scene.sh racing" >&2; exit 1; }
done

min_free="${MIN_FREE_MIB:-30000}"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
echo "GPU free: ${free_mib} MiB (needs at least ${min_free})"
if [[ "${free_mib}" -lt "${min_free}" ]]; then
  echo "ERROR: not enough free GPU memory; the GPU is shared. Not starting." >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
output="${output:-outputs/racing_${stamp}}"
# `matrix-game render` refuses to start when the output directory exists, so only the parent is created here
mkdir -p "$(dirname "${output}")"

if [[ -n "${take}" ]]; then
  source_args=(--take "${take}")
else
  if [[ -z "${input}" ]]; then
    input="${output}_synthetic_input.mp4"
    "${python_bin}" scripts/dgx/make_test_input.py --output "${input}" --seconds "${seconds}"
    echo "NOTE: ${input} is a synthetic stand-in, not a game recording."
  fi
  source_args=(--input "${input}")
fi

echo "Rendering racing (${preset}, fixed prompt) -> ${output}"
start=${SECONDS}
"${tool}" render --scene racing "${source_args[@]}" --resources resources --preset "${preset}" --prompt-mode fixed --output "${output}" "${extra[@]}" 2>&1 | tee "${output}.render.log"
echo "finished in $((SECONDS - start)) s. Output: ${output}/rendered.mp4 (and a configuration report); log: ${output}.render.log"
