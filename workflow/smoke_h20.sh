#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${IMAGENET_TRAIN:?Set IMAGENET_TRAIN}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

asset_root="${ASSET_ROOT:-$OUTPUT_ROOT/assets}"
conv_source="$asset_root/huggingface/BlueSourceJY/SiT-Complementary/checkpoints/bs256_lr1e-4/conv-layer/1950000.pt"
marker="$OUTPUT_ROOT/.h20_smoke_passed"
if [[ -f "$marker" ]]; then
    echo "H20_SMOKE_ALREADY_PASSED=$marker"
    exit 0
fi

smoke_dir="$(mktemp -d "$OUTPUT_ROOT/.h20-smoke-XXXXXX")"
cleanup() {
    if [[ "$smoke_dir" == "$OUTPUT_ROOT/".h20-smoke-* ]]; then
        rm -rf -- "$smoke_dir"
    fi
}
trap cleanup EXIT

common=(
    --model SiT-S/2
    --epochs 800
    --data-path "$IMAGENET_TRAIN"
    --global-batch-size 256
    --gradient-accumulation-steps 1
    --restart-deterministic-data
    --learning-rate 1e-4
    --num-workers 0
    --log-every 1
    --ckpt-every 999999999
    --sample-every 999999999
    --sample-batch-size 1
)

torchrun --standalone --nnodes=1 --nproc_per_node=8 train_conv.py \
    "${common[@]}" \
    --max-train-steps 1950001 \
    --results-dir "$smoke_dir/conv" \
    --run-name smoke-conv \
    --ckpt "$conv_source"
python workflow/checkpoint_tool.py inspect \
    "$smoke_dir/conv/smoke-conv/checkpoints/1950001.pt" | grep -qx 1950001

torchrun --standalone --nnodes=1 --nproc_per_node=8 train_rot_head.py \
    "${common[@]}" \
    --max-train-steps 1 \
    --results-dir "$smoke_dir/rotation-head" \
    --run-name smoke-rotation-head
python workflow/checkpoint_tool.py inspect \
    "$smoke_dir/rotation-head/smoke-rotation-head/checkpoints/0000001.pt" | grep -qx 1

touch "$marker"
echo "H20_SMOKE_PASS=$marker"
