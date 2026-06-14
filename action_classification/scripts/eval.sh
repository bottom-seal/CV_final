#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/diving48}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/results/full_finetune_videomaev2_base}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/outputs/runs/full_finetune_videomaev2_base/checkpoint-best.pth}"
GPUS="${GPUS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-35}"
LR="${LR:-0.0001}"
WANDB_PROJECT="${WANDB_PROJECT:-videomaev2-diving48}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-full_finetune_videomaev2_base}"

CUDA_VISIBLE_DEVICES="${GPUS}" python "${ROOT_DIR}/scripts/evaluate.py" \
  --checkpoint "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --run-name "${WANDB_RUN_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --wandb-project "${WANDB_PROJECT}" \
  "$@"
