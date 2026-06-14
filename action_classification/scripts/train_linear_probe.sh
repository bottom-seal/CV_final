#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/diving48}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/runs/linear_probe_videomaev2_base}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/checkpoints/videomaev2_base.pth}"
GPUS="${GPUS:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.001}"
WANDB_PROJECT="${WANDB_PROJECT:-videomaev2-diving48}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-linear_probe_videomaev2_base}"

export DATA_PATH OUTPUT_DIR MODEL_PATH GPUS BATCH_SIZE EPOCHS LR
export WANDB_PROJECT WANDB_RUN_NAME
export RUN_NAME="${WANDB_RUN_NAME}" FREEZE_MODE=linear DROP_PATH=0.0
export TEST_NUM_SEGMENT="${TEST_NUM_SEGMENT:-1}" TEST_NUM_CROP="${TEST_NUM_CROP:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec "${ROOT_DIR}/scripts/run_training.sh" "$@"
