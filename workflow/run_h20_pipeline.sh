#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source workflow/h20_common.sh

: "${IMAGENET_TRAIN:?Set IMAGENET_TRAIN to the ImageNet train ImageFolder}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to durable experiment storage}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export ASSET_ROOT="${ASSET_ROOT:-$OUTPUT_ROOT/assets}"
export SIT_VAE_ROOT="$ASSET_ROOT/vae"
export TORCH_HOME="$ASSET_ROOT/cache/torch"
export HF_HOME="$ASSET_ROOT/cache/huggingface"

prepare_mode="${PREPARE_ASSETS:-auto}"
if [[ "$prepare_mode" != "auto" && "$prepare_mode" != "0" && "$prepare_mode" != "1" ]]; then
    echo "PREPARE_ASSETS must be auto, 0, or 1" >&2
    exit 2
fi
if [[ "$prepare_mode" == "0" ]]; then
    export HF_HUB_OFFLINE=1
    python workflow/prepare_assets.py --asset-root "$ASSET_ROOT" --local-only
    python workflow/prewarm.py --asset-root "$ASSET_ROOT" --local-only
else
    unset HF_HUB_OFFLINE
    python workflow/prepare_assets.py --asset-root "$ASSET_ROOT"
    python workflow/prewarm.py --asset-root "$ASSET_ROOT"
fi
if ! h20_assets_ready "$ASSET_ROOT"; then
    echo "Asset preparation did not produce the complete pinned asset set" >&2
    exit 1
fi
export HF_HUB_OFFLINE=1
python workflow/preflight.py \
    --imagenet-train "$IMAGENET_TRAIN" \
    --output-root "$OUTPUT_ROOT" \
    --nproc "$NPROC_PER_NODE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --require-clean-git
bash workflow/smoke_h20.sh

export PREPARE_ASSETS=0
export SKIP_PREFLIGHT=1
export SKIP_SMOKE=1

conv_periodic_targets=(
    2000000 2250000 2500000 2750000 3000000
    3250000 3500000 3750000 4000000
)
rotation_head_periodic_targets=(
    250000 500000 750000 1000000 1250000 1500000 1750000 2000000
    2250000 2500000 2750000 3000000 3250000 3500000 3750000 4000000
)

# On a dedicated node each model trains in one continuous process. Periodic
# checkpoints are evaluated only after that model reaches its final step, so
# evaluation does not introduce artificial RNG restart boundaries.
TRAIN_ONLY=1 bash workflow/run_h20_stage.sh conv 4003200
for target in "${conv_periodic_targets[@]}"; do
    EVALUATE_ONLY=1 bash workflow/run_h20_stage.sh conv "$target"
done
EVALUATE_ONLY=1 bash workflow/run_h20_stage.sh conv 4003200

TRAIN_ONLY=1 bash workflow/run_h20_stage.sh rotation-head 4003200
for target in "${rotation_head_periodic_targets[@]}"; do
    EVALUATE_ONLY=1 bash workflow/run_h20_stage.sh rotation-head "$target"
done
EVALUATE_ONLY=1 bash workflow/run_h20_stage.sh rotation-head 4003200

echo "H20_PIPELINE_COMPLETE=$OUTPUT_ROOT/training_results/TRAINING_RESULTS.md"
