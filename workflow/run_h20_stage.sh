#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source workflow/h20_common.sh

variant="${1:-${PIPELINE_VARIANT:-}}"
target="${2:-${PIPELINE_TARGET:-}}"
if [[ "$variant" != "conv" && "$variant" != "rotation-head" ]]; then
    echo "Usage: $0 {conv|rotation-head} TARGET_STEP" >&2
    exit 2
fi
if [[ ! "$target" =~ ^[0-9]+$ ]]; then
    echo "TARGET_STEP must be an integer" >&2
    exit 2
fi

: "${IMAGENET_TRAIN:?Set IMAGENET_TRAIN to the ImageNet train ImageFolder}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to durable experiment storage}"

asset_root="${ASSET_ROOT:-$OUTPUT_ROOT/assets}"
result_root="$OUTPUT_ROOT/training_results"
training_root="$OUTPUT_ROOT/training"
nproc="${NPROC_PER_NODE:-8}"
global_batch="${GLOBAL_BATCH_SIZE:-256}"
accumulation="${GRADIENT_ACCUMULATION_STEPS:-1}"
final_step=4003200

if [[ "$nproc" != "8" || "$global_batch" != "256" || "$accumulation" != "1" ]]; then
    echo "This handoff requires NPROC_PER_NODE=8, GLOBAL_BATCH_SIZE=256, and GRADIENT_ACCUMULATION_STEPS=1" >&2
    exit 2
fi
if [[ "${TRAIN_ONLY:-0}" == "1" && "${EVALUATE_ONLY:-0}" == "1" ]]; then
    echo "TRAIN_ONLY and EVALUATE_ONLY are mutually exclusive" >&2
    exit 2
fi

export SIT_VAE_ROOT="$asset_root/vae"
export TORCH_HOME="$asset_root/cache/torch"
export HF_HOME="$asset_root/cache/huggingface"
conv_repo_dir="$asset_root/huggingface/BlueSourceJY/SiT-Complementary"
conv_source="$conv_repo_dir/checkpoints/bs256_lr1e-4/conv-layer/1950000.pt"
conv_history="$conv_repo_dir/experiments/bs256_lr1e-4/conv-layer/fid_cfg1_50k.tsv"
fid_reference="$asset_root/fid/VIRTUAL_imagenet256_labeled.npz"

if [[ "${TRAIN_ONLY:-0}" != "1" && "${EVALUATE_ONLY:-0}" != "1" ]] && \
    h20_stage_complete "$variant" "$target" "$OUTPUT_ROOT" "$asset_root"; then
    echo "H20_STAGE_ALREADY_COMPLETE variant=$variant target=$target"
    exit 0
fi

prepare_mode="${PREPARE_ASSETS:-auto}"
if [[ "$prepare_mode" != "auto" && "$prepare_mode" != "0" && "$prepare_mode" != "1" ]]; then
    echo "PREPARE_ASSETS must be auto, 0, or 1" >&2
    exit 2
fi
if [[ "$prepare_mode" == "0" ]]; then
    export HF_HUB_OFFLINE=1
    python workflow/prepare_assets.py --asset-root "$asset_root" --local-only
    python workflow/prewarm.py --asset-root "$asset_root" --local-only
else
    unset HF_HUB_OFFLINE
    python workflow/prepare_assets.py --asset-root "$asset_root"
    python workflow/prewarm.py --asset-root "$asset_root"
fi
if ! h20_assets_ready "$asset_root"; then
    echo "Asset preparation did not produce the complete pinned asset set" >&2
    exit 1
fi
export HF_HUB_OFFLINE=1

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    python workflow/preflight.py \
        --imagenet-train "$IMAGENET_TRAIN" \
        --output-root "$OUTPUT_ROOT" \
        --nproc "$nproc" \
        --global-batch-size "$global_batch" \
        --gradient-accumulation-steps "$accumulation" \
        --require-clean-git
fi

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    bash workflow/smoke_h20.sh
fi

if [[ "$variant" == "conv" ]]; then
    script="train_conv.py"
    run_name="SiT-S-2-ConvLayer-bs256-lr1e-4-800ep"
    first_target=2000000
    fallback=(--fallback "$conv_source")
else
    script="train_rot_head.py"
    run_name="SiT-S-2-RotationHead-bs256-lr1e-4-800ep"
    first_target=250000
    fallback=()
fi

if (( target < first_target || target > final_step )); then
    echo "Invalid $variant target: $target" >&2
    exit 2
fi
if (( target != final_step && target % 250000 != 0 )); then
    echo "Non-final targets must be divisible by 250000" >&2
    exit 2
fi

experiment_dir="$training_root/$run_name"
checkpoint_dir="$experiment_dir/checkpoints"
target_checkpoint="$checkpoint_dir/$(printf '%07d' "$target").pt"
mkdir -p "$checkpoint_dir" "$result_root"

keep_samples=()
if [[ "${KEEP_FID_SAMPLES:-0}" == "1" ]]; then
    keep_samples=(--keep-samples)
fi

evaluate() {
    local checkpoint="$1"
    local step="$2"
    local cfg="$3"
    python workflow/evaluate_checkpoint.py \
        --variant "$variant" \
        --checkpoint "$checkpoint" \
        --step "$step" \
        --cfg "$cfg" \
        --reference "$fid_reference" \
        --result-root "$result_root" \
        --nproc "$nproc" \
        "${keep_samples[@]}"
}

if [[ "${EVALUATE_ONLY:-0}" == "1" && ! -f "$target_checkpoint" ]]; then
    echo "Evaluate-only target checkpoint is missing: $target_checkpoint" >&2
    exit 1
fi

if [[ -f "$target_checkpoint" ]]; then
    actual_step="$(python workflow/checkpoint_tool.py inspect "$target_checkpoint")"
    if [[ "$actual_step" != "$target" ]]; then
        echo "Target checkpoint has the wrong step: $actual_step" >&2
        exit 1
    fi
    echo "TRAINING_ALREADY_COMPLETE=$target_checkpoint"
else
    resume_args=()
    if resume_checkpoint="$(python workflow/checkpoint_tool.py latest \
        --directory "$checkpoint_dir" --max-step "$target" "${fallback[@]}" 2>/dev/null)"; then
        resume_args=(--ckpt "$resume_checkpoint")
        echo "RESUME_CHECKPOINT=$resume_checkpoint"
    elif [[ "$variant" == "conv" ]]; then
        echo "No valid Conv resume checkpoint was found" >&2
        exit 1
    else
        echo "Starting Rotation-head from scratch"
    fi

    wandb_args=()
    if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_DISABLED:-0}" != "1" ]]; then
        wandb_args=(
            --wandb
            --wandb-project "${WANDB_PROJECT:-SiT-Complementary}"
            --run-name "$run_name"
        )
        if [[ -n "${WANDB_ENTITY:-}" ]]; then
            wandb_args+=(--wandb-entity "$WANDB_ENTITY")
        fi
    fi

    torchrun --standalone --nnodes=1 --nproc_per_node="$nproc" "$script" \
        --model SiT-S/2 \
        --epochs 800 \
        --max-train-steps "$target" \
        --data-path "$IMAGENET_TRAIN" \
        --results-dir "$training_root" \
        --global-batch-size "$global_batch" \
        --gradient-accumulation-steps "$accumulation" \
        --restart-deterministic-data \
        --learning-rate 1e-4 \
        --num-workers "${NUM_WORKERS_PER_GPU:-4}" \
        --log-every 100 \
        --ckpt-every 50000 \
        --sample-every 10000 \
        --cfg-scale 4.0 \
        --sample-batch-size 32 \
        --run-name "$run_name" \
        "${resume_args[@]}" \
        "${wandb_args[@]}"

    actual_step="$(python workflow/checkpoint_tool.py inspect "$target_checkpoint")"
    if [[ "$actual_step" != "$target" ]]; then
        echo "Training did not produce the expected checkpoint step" >&2
        exit 1
    fi
fi

if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
    echo "H20_TRAINING_COMPLETE variant=$variant target=$target"
    exit 0
fi

# Establish a measured resume baseline and merge it with the downloaded
# 250k..1.75m Conv history in the final report.
if [[ "$variant" == "conv" && "$target" == "$first_target" ]]; then
    evaluate "$conv_source" 1950000 1
fi

if (( target % 250000 == 0 )); then
    evaluate "$target_checkpoint" "$target" 1
fi
if (( target == final_step )); then
    evaluate "$target_checkpoint" "$target" 1
    evaluate "$target_checkpoint" "$target" 4
fi

if [[ ! -f "$conv_history" ]]; then
    echo "Pinned Conv FID history is missing: $conv_history" >&2
    exit 1
fi
build_args=(--output-dir "$result_root" --conv-history "$conv_history")
python workflow/build_results.py "${build_args[@]}"
if [[ "$variant" == "rotation-head" && "$target" == "$final_step" ]]; then
    python workflow/build_results.py "${build_args[@]}" --strict
fi

if ! h20_stage_complete "$variant" "$target" "$OUTPUT_ROOT" "$asset_root"; then
    echo "Stage artifacts did not pass the completion check" >&2
    python workflow/stage_status.py \
        --variant "$variant" \
        --target "$target" \
        --output-root "$OUTPUT_ROOT" \
        --asset-root "$asset_root"
    exit 1
fi

echo "H20_STAGE_COMPLETE variant=$variant target=$target"
