#!/usr/bin/env bash
# Matrix-Game 4.0 setup for the DGX (aarch64, GB300, CUDA 13.2): creates .venv with uv on Python 3.12 (the README's tested
# version), installs the pinned torch 2.12.0 / torchvision 0.27.0 from the PyTorch CUDA index first, then the package with its
# inference, server, download and test extras, and stops unless torch can run a bf16 matmul on the GPU.
#
# pyproject.toml pins torch==2.12.0, which pip/uv would otherwise resolve to a build that does not match this machine's CUDA, so
# the torch build is installed explicitly first and satisfies the pin afterwards.
#
# Usage:   bash scripts/dgx/setup_dgx.sh                    (run from anywhere; safe to run again)
#   CUDA_TAG=cu130 bash scripts/dgx/setup_dgx.sh            another PyTorch CUDA build; the default cu132 is CUDA 13.2
#   PYTHON_VERSION=3.11 bash scripts/dgx/setup_dgx.sh       another Python (pyproject allows 3.11 and 3.12)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

CUDA_TAG="${CUDA_TAG:-cu132}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
venv_python=".venv/bin/python"

export PATH="${HOME}/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found. Install it first: https://docs.astral.sh/uv/" >&2; exit 1; }

# the Windows checkout of this repository has CRLF line endings in its shell scripts, which bash rejects
# ("set: pipefail: invalid option name"); strip them from the shell scripts of this copy
for script in scripts/*.sh scripts/dgx/*.sh; do
  sed -i 's/\r$//' "${script}"
done

if [[ ! -x "${venv_python}" ]]; then
  echo "Creating venv with Python ${PYTHON_VERSION}..."
  uv venv --python "${PYTHON_VERSION}" .venv
fi

check_cuda() {
  "${venv_python}" -c '
import sys
import torch
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda, "| available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(1)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite((x @ x).float()).all()
print("GPU:", torch.cuda.get_device_name(0), "| bf16 matmul ok")
' || { echo "ERROR: torch cannot use the GPU. Stopping." >&2; exit 1; }
}

echo "===== torch 2.12.0 / torchvision 0.27.0 (${CUDA_TAG}) ====="
uv pip install --python "${venv_python}" torch==2.12.0 torchvision==0.27.0 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
check_cuda

echo "===== matrix-game with the inference, server, download and test extras ====="
uv pip install --python "${venv_python}" -e ".[inference,server,download,test]"
check_cuda

if [[ -d /usr/local/cuda-13.2 ]]; then
  echo "CUDA toolkit 13.2 found at /usr/local/cuda-13.2 (only needed to build SageAttention for the rtx5090-sage preset)."
fi
echo
echo "=========================================="
echo "Done. Venv: .venv (Python ${PYTHON_VERSION}, torch ${CUDA_TAG}). Next: bash scripts/dgx/download_scene.sh racing"
echo "=========================================="
