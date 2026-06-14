#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/diving48}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/runs/random_init_vit_base}"
MODEL_PATH=""
GPUS="${GPUS:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-0.0003}"
WANDB_PROJECT="${WANDB_PROJECT:-videomaev2-diving48}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-random_init_vit_base}"

export DATA_PATH OUTPUT_DIR MODEL_PATH GPUS BATCH_SIZE EPOCHS LR
export WANDB_PROJECT WANDB_RUN_NAME
export RUN_NAME="${WANDB_RUN_NAME}" FREEZE_MODE=full
export WANDB_MODE="${WANDB_MODE:-online}"
exec "${ROOT_DIR}/scripts/run_training.sh" "$@"
