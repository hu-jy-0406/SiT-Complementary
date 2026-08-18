#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to the ImageNet ImageFolder training directory}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS="${NUM_GPUS:-8}"
EPOCHS="${EPOCHS:-800}"
RESULTS_DIR="${RESULTS_DIR:-results/base}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
RUN_NAME="${RUN_NAME:-SiT-S-2-Base-bs256-lr1e-4}"
CKPT="${CKPT:-}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
ENABLE_FID="${ENABLE_FID:-0}"
TRAINING_ENTRYPOINT="${TRAINING_ENTRYPOINT:-train.py}"

extra_args=()
if [[ -n "$CKPT" ]]; then
  extra_args+=(--ckpt "$CKPT")
fi
if [[ "$ENABLE_WANDB" == "1" ]]; then
  extra_args+=(--wandb)
fi
if [[ "$ENABLE_FID" == "1" ]]; then
  : "${FID_REFERENCE:?Set FID_REFERENCE when ENABLE_FID=1}"
  extra_args+=(
    --fid-every-checkpoint
    --fid-every "${FID_EVERY:-250000}"
    --fid-num-samples "${FID_NUM_SAMPLES:-50000}"
    --fid-reference "$FID_REFERENCE"
    --fid-history "${FID_HISTORY:-$RESULTS_DIR/$RUN_NAME/fid_cfg1_50k.tsv}"
    --fid-per-proc-batch-size "${FID_PER_PROC_BATCH_SIZE:-64}"
    --fid-inception-batch-size "${FID_INCEPTION_BATCH_SIZE:-128}"
    --fid-num-workers "${FID_NUM_WORKERS:-8}"
    --fid-sampling-steps "${FID_SAMPLING_STEPS:-250}"
    --fid-seed "${FID_SEED:-0}"
  )
fi

train_args=(
  --model SiT-S/2
  --epochs "$EPOCHS"
  --data-path "$DATA_PATH"
  --results-dir "$RESULTS_DIR"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --global-seed "${GLOBAL_SEED:-0}"
  --vae "${VAE:-ema}"
  --num-workers "${NUM_WORKERS:-4}"
  --log-every "${LOG_EVERY:-100}"
  --ckpt-every "${CKPT_EVERY:-50000}"
  --sample-every "${SAMPLE_EVERY:-10000}"
  --cfg-scale "${CFG_SCALE:-4.0}"
  --run-name "$RUN_NAME"
)
train_args+=("$@")
train_args+=("${extra_args[@]}")

exec torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  "$TRAINING_ENTRYPOINT" "${train_args[@]}"
