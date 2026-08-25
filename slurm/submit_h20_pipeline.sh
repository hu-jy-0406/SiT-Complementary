#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source workflow/h20_common.sh

: "${IMAGENET_TRAIN:?Set IMAGENET_TRAIN before submitting}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT before submitting}"
asset_root="${ASSET_ROOT:-$OUTPUT_ROOT/assets}"
export GPU_PROFILE="${GPU_PROFILE:-a100-40gb}"
: "${EXPERIMENT:?Set EXPERIMENT to conv, rotation-head, or all}"
experiment="$EXPERIMENT"
if [[ "$experiment" != "conv" && "$experiment" != "rotation-head" && "$experiment" != "all" ]]; then
    echo "EXPERIMENT must be conv, rotation-head, or all" >&2
    exit 2
fi
handoff_validate_gpu_profile "$GPU_PROFILE"
handoff_lock_gpu_profile "$OUTPUT_ROOT" "$GPU_PROFILE"

partition_args=()
[[ -n "${SLURM_PARTITION:-}" ]] && partition_args+=(--partition "$SLURM_PARTITION")
[[ -n "${SLURM_ACCOUNT:-}" ]] && partition_args+=(--account "$SLURM_ACCOUNT")
[[ -n "${SLURM_QOS:-}" ]] && partition_args+=(--qos "$SLURM_QOS")
[[ -n "${SLURM_TIME:-}" ]] && partition_args+=(--time "$SLURM_TIME")
[[ -n "${SLURM_NODELIST:-}" ]] && partition_args+=(--nodelist "$SLURM_NODELIST")
default_gres="$(handoff_default_slurm_gres "$GPU_PROFILE")"
gres="${SLURM_GRES:-$default_gres}"

previous="${SLURM_AFTEROK_JOB_ID:-}"
if [[ -n "$previous" && ! "$previous" =~ ^[0-9]+$ ]]; then
    echo "SLURM_AFTEROK_JOB_ID must be a numeric Slurm job ID" >&2
    exit 2
fi
prepare_next=0
if ! h20_assets_ready "$asset_root"; then
    prepare_next=1
fi
submitted=0

parse_job_id() {
    local output="$1"
    local line=""
    while IFS= read -r line; do
        line="${line%$'\r'}"
        if [[ "$line" =~ ^([0-9]+)(\;.*)?$ ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
            return 0
        fi
        if [[ "$line" =~ ^Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+)$ ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
            return 0
        fi
    done <<< "$output"
    return 1
}

submit_stage() {
    local variant="$1"
    local target="$2"
    if h20_stage_complete "$variant" "$target" "$OUTPUT_ROOT" "$asset_root"; then
        local finalize_marker="$OUTPUT_ROOT/.finalize_passed-$variant"
        if [[ "$target" != "4003200" || -f "$finalize_marker" ]]; then
            echo "SKIPPED_COMPLETE variant=$variant target=$target"
            return 0
        fi
        echo "SUBMITTING_FINALIZER variant=$variant target=$target"
    fi

    local dependency_args=()
    [[ -n "$previous" ]] && dependency_args=(--dependency "afterok:$previous")
    local prepare=0
    if (( prepare_next )); then
        prepare=1
    fi
    local submission=""
    submission="$(sbatch --parsable \
        --gres "$gres" \
        "${partition_args[@]}" \
        "${dependency_args[@]}" \
        --export="ALL,GPU_PROFILE=$GPU_PROFILE,PIPELINE_VARIANT=$variant,PIPELINE_TARGET=$target,PREPARE_ASSETS=$prepare" \
        slurm/h20_stage.slurm)"
    local job_id=""
    if ! job_id="$(parse_job_id "$submission")"; then
        echo "Could not parse sbatch --parsable output: $submission" >&2
        exit 1
    fi
    previous="$job_id"
    prepare_next=0
    submitted=$((submitted + 1))
    echo "SUBMITTED variant=$variant target=$target job=$previous"
}

if [[ "$experiment" == "conv" || "$experiment" == "all" ]]; then
    for target in "${H20_CONV_TARGETS[@]}"; do
        submit_stage conv "$target"
    done
fi
if [[ "$experiment" == "rotation-head" || "$experiment" == "all" ]]; then
    for target in "${H20_ROTATION_HEAD_TARGETS[@]}"; do
        submit_stage rotation-head "$target"
    done
fi

if (( submitted )); then
    echo "SUBMITTED_STAGE_COUNT=$submitted"
    echo "FINAL_JOB_ID=$previous"
else
    echo "NO_STAGES_SUBMITTED=all required stages are already complete"
fi
