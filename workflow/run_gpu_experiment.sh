#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source workflow/h20_common.sh

variant="${1:-${EXPERIMENT:-}}"
if [[ "$variant" != "conv" && "$variant" != "rotation-head" ]]; then
    echo "Usage: $0 {conv|rotation-head}" >&2
    exit 2
fi
: "${IMAGENET_TRAIN:?Set IMAGENET_TRAIN to the ImageNet train ImageFolder}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to shared durable experiment storage}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export GPU_PROFILE="${GPU_PROFILE:-a100-40gb}"
export ASSET_ROOT="${ASSET_ROOT:-$OUTPUT_ROOT/assets}"
export SIT_VAE_ROOT="$ASSET_ROOT/vae"
export TORCH_HOME="$ASSET_ROOT/cache/torch"
export HF_HOME="$ASSET_ROOT/cache/huggingface"
handoff_validate_gpu_profile "$GPU_PROFILE"
handoff_lock_gpu_profile "$OUTPUT_ROOT" "$GPU_PROFILE"

prepare_args=()
if [[ "${PREPARE_ASSETS:-auto}" == "0" ]]; then
    prepare_args=(--local-only)
elif [[ "${PREPARE_ASSETS:-auto}" != "auto" && "${PREPARE_ASSETS:-auto}" != "1" ]]; then
    echo "PREPARE_ASSETS must be auto, 0, or 1" >&2
    exit 2
fi
python workflow/prepare_runtime.py --asset-root "$ASSET_ROOT" "${prepare_args[@]}"
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
    --gpu-profile "$GPU_PROFILE" \
    --require-clean-git
SMOKE_VARIANT="$variant" bash workflow/smoke_h20.sh

export PREPARE_ASSETS=0
export SKIP_ASSET_VERIFY=1
export SKIP_PREFLIGHT=1
export SKIP_SMOKE=1

if [[ "$variant" == "conv" ]]; then
    periodic_targets=(
        2000000 2250000 2500000 2750000 3000000
        3250000 3500000 3750000 4000000
    )
else
    periodic_targets=(
        250000 500000 750000 1000000 1250000 1500000 1750000 2000000
        2250000 2500000 2750000 3000000 3250000 3500000 3750000 4000000
    )
fi

# Train continuously to the exact 800-epoch step, then evaluate saved periodic
# checkpoints. This avoids introducing artificial RNG restart boundaries.
TRAIN_ONLY=1 bash workflow/run_h20_stage.sh "$variant" 4003200
for target in "${periodic_targets[@]}"; do
    EVALUATE_ONLY=1 bash workflow/run_h20_stage.sh "$variant" "$target"
done
EVALUATE_ONLY=1 bash workflow/run_h20_stage.sh "$variant" 4003200

echo "GPU_EXPERIMENT_COMPLETE=$variant"
