#!/usr/bin/env bash
#SBATCH --job-name=sit-rot-head
#SBATCH --partition=long
#SBATCH --exclude=gpu-12,gpu-24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-23:00:00
#SBATCH --output=slurm/rot-head-%j.out
#SBATCH --error=slurm/rot-head-%j.err

set -euo pipefail

cd /home/guangyi.chen/workspace/jinyuan/SiT-Complementary
source /home/guangyi.chen/miniconda3/etc/profile.d/conda.sh
conda activate sit-complementary

: "${WANDB_API_KEY:?Pass WANDB_API_KEY in the Slurm job environment}"
: "${SEGMENT_TARGET:?Pass the optimizer-step target for this segment}"

imagenet="/l/users/guangyi.chen/dataset/imagenet/ILSVRC/Data/CLS-LOC/train"
result_root="${RESULTS_DIR:-results-rot-head-4xa100-40g}"
run_name="${RUN_NAME:-SiT-S-2-RotationHead-bs256-lr1e-4-800ep}"
final_step=4003200
early_stop_marker="evaluation/fid_results/rot-head/EARLY_STOP"

if [[ -f "$early_stop_marker" ]]; then
    echo "Rotation-head training already met the historical FID early-stop rule: $early_stop_marker"
    exit 0
fi

resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    resume_args=(--ckpt "$RESUME_CHECKPOINT")
    echo "Resuming checkpoint: $RESUME_CHECKPOINT"
else
    echo "Starting Rotation-head training from scratch"
fi

torchrun --standalone --nnodes=1 --nproc_per_node=4 train_rot_head.py \
    --model SiT-S/2 \
    --epochs 800 \
    --max-train-steps "$SEGMENT_TARGET" \
    --data-path "$imagenet" \
    --results-dir "$result_root" \
    --global-batch-size 256 \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
    --learning-rate 1e-4 \
    --num-workers 4 \
    --log-every 100 \
    --ckpt-every 50000 \
    --sample-every 10000 \
    --cfg-scale 4.0 \
    --sample-batch-size "${SAMPLE_BATCH_SIZE:-32}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT:-SiT-Complementary}" \
    --run-name "$run_name" \
    "${resume_args[@]}"

checkpoint="$result_root/$run_name/checkpoints/$(printf '%07d' "$SEGMENT_TARGET").pt"
if [[ ! -f "$checkpoint" ]]; then
    echo "Expected checkpoint not found: $checkpoint" >&2
    exit 1
fi

# Match the historical Base/Rotation-layer evaluation schedule and protocol.
if (( SEGMENT_TARGET % 250000 == 0 )); then
    CHECKPOINT="$checkpoint" STEP="$SEGMENT_TARGET" CFG=1.0 RUN_NAME="$run_name" \
        bash evaluation/evaluate_rot_head_checkpoint.sh
fi
if (( SEGMENT_TARGET >= final_step )); then
    CHECKPOINT="$checkpoint" STEP="$SEGMENT_TARGET" CFG=4.0 RUN_NAME="$run_name" \
        bash evaluation/evaluate_rot_head_checkpoint.sh
fi
