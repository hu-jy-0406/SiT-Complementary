#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to a Rotation-head training checkpoint}"
: "${STEP:?Set STEP to the checkpoint optimizer step}"

CFG="${CFG:-1.0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-64}"
SAMPLE_ROOT="${SAMPLE_ROOT:-evaluation/generated_rot_head_training}"
REFERENCE="${REFERENCE:-evaluation/VIRTUAL_imagenet256_labeled.npz}"
RESULT_DIR="${RESULT_DIR:-evaluation/fid_results/rot-head}"
RUN_NAME="${RUN_NAME:-SiT-S-2-RotationHead-bs256-lr1e-4-800ep}"
WANDB_PROJECT="${WANDB_PROJECT:-SiT-Complementary}"

checkpoint_name="$(basename "$CHECKPOINT" .pt)"
folder="$SAMPLE_ROOT/SiT-S-2-rot-head-${checkpoint_name}-cfg-${CFG}-${PER_PROC_BATCH_SIZE}-ODE-250-euler"
result="$RESULT_DIR/step-${STEP}-cfg-${CFG}.txt"
tsv="$RESULT_DIR/fid_results.tsv"

mkdir -p "$SAMPLE_ROOT" "$RESULT_DIR"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" sample_ddp.py ODE \
    --variant rot-head \
    --model SiT-S/2 \
    --num-fid-samples 50000 \
    --keep-padded-samples \
    --cfg-scale "$CFG" \
    --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
    --num-sampling-steps 250 \
    --sampling-method euler \
    --global-seed 0 \
    --sample-dir "$SAMPLE_ROOT" \
    --ckpt "$CHECKPOINT"

pytorch-fid "$folder" "$REFERENCE" \
    --device cuda:0 --batch-size 128 | tee "$result"

fid="$(awk '/FID:/ {value=$NF} END {print value}' "$result")"
if [[ -z "$fid" ]]; then
    echo "Could not parse FID from $result" >&2
    exit 1
fi

if [[ ! -f "$tsv" ]]; then
    printf 'step\tcheckpoint\tstatus\tfid\tcfg\tnum_requested\tnum_png\tseed\ttimestamp_utc\n' > "$tsv"
fi
num_png="$(find "$folder" -maxdepth 1 -type f -name '*.png' | wc -l)"
timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if ! awk -F '\t' -v step="$STEP" -v cfg="$CFG" \
    'NR > 1 && $1 == step && ($5 + 0) == (cfg + 0) { found=1 } END { exit !found }' \
    "$tsv"; then
    printf '%s\t%s\tok\t%s\t%s\t50000\t%s\t0\t%s\n' \
        "$STEP" "$CHECKPOINT" "$fid" "$CFG" "$num_png" "$timestamp_utc" >> "$tsv"
else
    echo "FID history already contains step=$STEP cfg=$CFG; not appending a duplicate"
fi

if [[ "$CFG" == "1" || "$CFG" == "1.0" ]]; then
    python evaluation/check_fid_early_stop.py \
        --history "$tsv" \
        --marker "$RESULT_DIR/EARLY_STOP"
fi

if [[ "${WANDB_LOG_FID:-1}" == "1" ]]; then
    : "${WANDB_API_KEY:?Set WANDB_API_KEY to log FID without wandb login}"
    python evaluation/log_fid_wandb.py \
        --run-name "$RUN_NAME" \
        --project "$WANDB_PROJECT" \
        ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} \
        --step "$STEP" \
        --cfg "$CFG" \
        --fid "$fid"
fi

echo "ROT_HEAD_FID_COMPLETE step=$STEP cfg=$CFG fid=$fid num_png=$num_png"
