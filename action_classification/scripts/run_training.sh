#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:?DATA_PATH is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${FREEZE_MODE:?FREEZE_MODE is required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-vit_base_patch16_224}"
MODEL_PATH="${MODEL_PATH:-}"
GPUS="${GPUS:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
EPOCHS="${EPOCHS:-35}"
LR="${LR:-0.0001}"
WANDB_PROJECT="${WANDB_PROJECT:-videomaev2-diving48}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_FRAMES="${NUM_FRAMES:-16}"
SAMPLING_RATE="${SAMPLING_RATE:-4}"
TEST_NUM_SEGMENT="${TEST_NUM_SEGMENT:-5}"
TEST_NUM_CROP="${TEST_NUM_CROP:-3}"
UNFREEZE_LAST_BLOCKS="${UNFREEZE_LAST_BLOCKS:-4}"
DROP_PATH="${DROP_PATH:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LAYER_DECAY="${LAYER_DECAY:-0.8}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_LR="${WARMUP_LR:-1e-8}"
CLIP_GRAD="${CLIP_GRAD:-5.0}"
NUM_SAMPLE="${NUM_SAMPLE:-1}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-1}"
SEED="${SEED:-42}"

if [[ ! -f "${DATA_PATH}/train.csv" || ! -f "${DATA_PATH}/val.csv" ]]; then
  echo "Missing ${DATA_PATH}/train.csv or val.csv." >&2
  echo "Run: python scripts/prepare_diving48.py --strict" >&2
  exit 2
fi
if [[ -n "${MODEL_PATH}" && ! -f "${MODEL_PATH}" ]]; then
  echo "Missing checkpoint: ${MODEL_PATH}" >&2
  echo "Run: python scripts/download_checkpoint.py --model base" >&2
  exit 2
fi
if [[ "${BATCH_SIZE}" == "auto" ]]; then
  GPU_MEMORY="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    | head -n 1 || true)"
  if [[ -n "${GPU_MEMORY}" && "${GPU_MEMORY}" -ge 22000 ]]; then
    BATCH_SIZE=8
  else
    BATCH_SIZE=4
  fi
fi

mkdir -p "${OUTPUT_DIR}"
FINETUNE_ARGS=()
if [[ -n "${MODEL_PATH}" ]]; then
  FINETUNE_ARGS=(--finetune "${MODEL_PATH}")
fi

cd "${ROOT_DIR}/VideoMAEv2"
CUDA_VISIBLE_DEVICES="${GPUS}" python run_class_finetuning.py \
  --model "${MODEL}" \
  --data_set Diving48 \
  --nb_classes 48 \
  --data_path "${DATA_PATH}" \
  --data_root "${DATA_PATH}/videos" \
  "${FINETUNE_ARGS[@]}" \
  --freeze_mode "${FREEZE_MODE}" \
  --unfreeze_last_blocks "${UNFREEZE_LAST_BLOCKS}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${OUTPUT_DIR}/tensorboard" \
  --batch_size "${BATCH_SIZE}" \
  --update_freq "${UPDATE_FREQ}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --no_lr_scale \
  --min_lr "${MIN_LR}" \
  --warmup_lr "${WARMUP_LR}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --input_size 224 \
  --short_side_size 224 \
  --num_frames "${NUM_FRAMES}" \
  --sampling_rate "${SAMPLING_RATE}" \
  --num_sample "${NUM_SAMPLE}" \
  --num_workers "${NUM_WORKERS}" \
  --opt adamw \
  --opt_betas 0.9 0.999 \
  --weight_decay "${WEIGHT_DECAY}" \
  --drop_path "${DROP_PATH}" \
  --clip_grad "${CLIP_GRAD}" \
  --layer_decay "${LAYER_DECAY}" \
  --test_num_segment "${TEST_NUM_SEGMENT}" \
  --test_num_crop "${TEST_NUM_CROP}" \
  --save_ckpt_freq "${SAVE_CKPT_FREQ}" \
  --seed "${SEED}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --wandb_mode "${WANDB_MODE}" \
  "$@"
