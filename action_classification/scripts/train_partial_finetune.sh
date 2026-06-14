#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/diving48}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/runs/partial_finetune_videomaev2_base}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/videomaev2_base.pth}"
GPUS="${GPUS:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-0.0002}"
WANDB_PROJECT="${WANDB_PROJECT:-videomaev2-diving48}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-partial_finetune_videomaev2_base}"

export DATA_PATH OUTPUT_DIR MODEL_PATH GPUS BATCH_SIZE EPOCHS LR
export WANDB_PROJECT WANDB_RUN_NAME
export RUN_NAME="${WANDB_RUN_NAME}" FREEZE_MODE=partial
export UNFREEZE_LAST_BLOCKS="${UNFREEZE_LAST_BLOCKS:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec "${ROOT_DIR}/scripts/run_training.sh" "$@"
