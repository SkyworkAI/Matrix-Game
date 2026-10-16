#!/usr/bin/env bash
# Download and verify the Matrix-Game 4.0 resources of one scene (or all) at the pinned release revision: the scene's game assets and
# model checkpoint plus the shared encoder/codec files. The shared T5 encoder and Wan2.2 VAE are taken from the Hugging Face cache when a
# copy with the manifest's SHA256 already exists (see link_cached_base.py), otherwise they are downloaded. Resumable and safe to run again:
# files that already verify are skipped. Needs scripts/dgx/setup_dgx.sh to have completed.
#
# Sizes: the shared files are 14.2 GB, each scene checkpoint is 10 GB (castle, ancient-egypt, racing; 44.2 GB in all), the game assets about
# 190 MB. Allow room for the Hugging Face cache as well.
#
# Usage:   bash scripts/dgx/download_scene.sh <racing|castle|ancient-egypt|all>
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

scene="${1:-}"
case "${scene}" in
  racing|castle|ancient-egypt|all) ;;
  *) echo "Usage: bash scripts/dgx/download_scene.sh <racing|castle|ancient-egypt|all>" >&2; exit 2 ;;
esac

revision="4419accdb7bb7881234c5914de8cd915266e0325"
repo="Skywork/Matrix-Game-4.0"
tool=".venv/bin/matrix-game"
[[ -x "${tool}" ]] || { echo "ERROR: ${tool} not found. Run scripts/dgx/setup_dgx.sh first." >&2; exit 1; }

echo "free disk here: $(df -h . | awk 'NR==2 {print $4}')"
.venv/bin/python scripts/dgx/link_cached_base.py --resources resources

for manifest in game-assets models; do
  echo "===== ${manifest} (${scene}) ====="
  "${tool}" download --manifest "releases/${manifest}.json" --repo-id "${repo}" --repo-type model --revision "${revision}" --scene "${scene}" --resources resources
  "${tool}" verify --manifest "releases/${manifest}.json" --scene "${scene}" --resources resources > /dev/null
  echo "${manifest}: verified"
done

echo
echo "Installed under resources/ ($(du -sh resources | cut -f1)). Next: bash scripts/dgx/run_racing.sh"
