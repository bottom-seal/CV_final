#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

"${PYTHON}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "Environment ready. Activate it with:"
echo "  source ${VENV_PATH}/bin/activate"
echo
echo "Optional multi-GPU/DeepSpeed support:"
echo "  python -m pip install deepspeed"

