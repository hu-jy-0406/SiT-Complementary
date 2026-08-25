#!/usr/bin/env bash
#SBATCH --job-name=sit-conv-resume-bench
#SBATCH --partition=long
#SBATCH --exclude=gpu-12,gpu-24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/conv-resume-bench-%j.out
#SBATCH --error=slurm/conv-resume-bench-%j.err

set -euo pipefail

cd /home/guangyi.chen/workspace/jinyuan/SiT-Complementary
source /home/guangyi.chen/miniconda3/etc/profile.d/conda.sh
conda activate sit-complementary

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total \
    --format=csv,noheader

checkpoint="pretrained_models/BlueSourceJY-SiT-Complementary/checkpoints/bs256_lr1e-4/conv-layer/1950000.pt"
imagenet="/l/users/guangyi.chen/dataset/imagenet/ILSVRC/Data/CLS-LOC/train"

run_benchmark() {
    local accumulation_steps="$1"
    local label="$2"

    echo "BENCHMARK_BEGIN:${label}"
    torchrun --standalone --nnodes=1 --nproc_per_node=2 train_conv.py \
        --model SiT-S/2 \
        --epochs 800 \
        --max-train-steps 1950001 \
        --data-path "$imagenet" \
        --results-dir "/tmp/sit-conv-resume-bench-${SLURM_JOB_ID}-${label}" \
        --global-batch-size 128 \
        --gradient-accumulation-steps "$accumulation_steps" \
        --learning-rate 1e-4 \
        --num-workers 4 \
        --log-every 1 \
        --ckpt-every 999999999 \
        --sample-every 999999999 \
        --ckpt "$checkpoint"
    echo "BENCHMARK_PASS:${label}"
}

run_benchmark 1 micro64-accum1
run_benchmark 2 micro32-accum2
