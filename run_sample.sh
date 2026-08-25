#!/usr/bin/env bash
set -euo pipefail

: "${CKPT:?Set CKPT to a trained checkpoint path}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" sample_ddp.py ODE \
  --variant "${VARIANT:-base}" \
  --model "${MODEL:-SiT-S/2}" \
  --num-fid-samples "${NUM_FID_SAMPLES:-50000}" \
  --cfg-scale "${CFG_SCALE:-4.0}" \
  --ckpt "$CKPT" \
  "$@"
