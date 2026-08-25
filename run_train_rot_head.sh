#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
: "${IMAGENET_PATH:?Set IMAGENET_PATH to the ImageNet train ImageFolder}"

torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" train_rot_head.py \
  --model "${MODEL:-SiT-S/2}" \
  --epochs "${EPOCHS:-800}" \
  --data-path "$IMAGENET_PATH" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  "$@"
