#!/usr/bin/env bash
# Render the City Racing scene with a custom sink image instead of the stock gray-sedan reference.
# The scene's absolute-position sink is written to output frame zero without denoising (see docs/racing.md),
# so the car's color/identity comes entirely from the sink image, not the text prompt -- the default
# resources/models/racing/sink.png shows a gray sedan, which is why prompts like "bright yellow sedan"
# never change the car's color on their own. This passes a yellow-taxi reference photo as the sink instead.
# Thin wrapper around run_racing.sh; same prerequisites (scripts/dgx/setup_dgx.sh, download_scene.sh racing).
#
# Usage:   bash scripts/dgx/run_racing_yellow_taxi.sh [--input lowpoly.mp4 | --take captures/take-001]
#                                                      [--sink-image path.jpg] [--seconds N] [--preset realtime|quality] [--output DIR]
#   --sink-image P   default: outputs/racing_take001_hq/Gemini_Generated_Image_e3sd2ce3sd2ce3sd.jpg
#   (all other flags are passed straight through to run_racing.sh -- see its --help)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

sink_image="outputs/racing_take001_hq/Gemini_Generated_Image_e3sd2ce3sd2ce3sd.jpg"
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sink-image) sink_image="${2:?--sink-image needs a file}"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) passthrough+=("$1"); shift ;;
  esac
done

[[ -f "${sink_image}" ]] || { echo "ERROR: sink image not found: ${sink_image}" >&2; exit 1; }

bash scripts/dgx/run_racing.sh "${passthrough[@]}" -- --sink-image "${sink_image}"
