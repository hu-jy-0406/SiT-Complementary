#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
: "${IMAGENET_PATH:?Set IMAGENET_PATH to the ImageNet train ImageFolder}"

torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" train.py \
  --model "${MODEL:-SiT-S/2}" \
  --epochs "${EPOCHS:-400}" \
  --data-path "$IMAGENET_PATH" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  "$@"
