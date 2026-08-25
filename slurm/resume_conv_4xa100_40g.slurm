#!/usr/bin/env bash
#SBATCH --job-name=sit-conv-resume
#SBATCH --partition=long
#SBATCH --exclude=gpu-12,gpu-24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-23:00:00
#SBATCH --output=slurm/conv-resume-%j.out
#SBATCH --error=slurm/conv-resume-%j.err

set -euo pipefail

cd /home/guangyi.chen/workspace/jinyuan/SiT-Complementary
source /home/guangyi.chen/miniconda3/etc/profile.d/conda.sh
conda activate sit-complementary

: "${WANDB_API_KEY:?Pass WANDB_API_KEY in the Slurm job environment}"

imagenet="/l/users/guangyi.chen/dataset/imagenet/ILSVRC/Data/CLS-LOC/train"
result_root="results-conv-resume-4xa100-40g"
run_name="SiT-S-2-ConvLayer-bs256-lr1e-4-800ep-resume-4xA100-40G"
final_step=4003200

: "${RESUME_CHECKPOINT:?Pass the exact input checkpoint path}"
: "${SEGMENT_TARGET:?Pass the optimizer-step target for this segment}"
checkpoint="$RESUME_CHECKPOINT"
segment_target="$SEGMENT_TARGET"

echo "Resuming checkpoint: $checkpoint"
echo "Training segment target: $segment_target (final: $final_step)"

torchrun --standalone --nnodes=1 --nproc_per_node=4 train_conv.py \
    --model SiT-S/2 \
    --epochs 800 \
    --max-train-steps "$segment_target" \
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
    --ckpt "$checkpoint" \
    --wandb \
    --wandb-project "${WANDB_PROJECT:-SiT-Complementary}" \
    --run-name "$run_name"

if (( segment_target >= final_step )); then
    echo "Training complete at step $final_step."
else
    echo "Segment complete at step $segment_target."
fi
