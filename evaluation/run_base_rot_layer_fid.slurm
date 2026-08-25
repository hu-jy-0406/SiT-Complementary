#!/usr/bin/env bash
#SBATCH --job-name=sit-fid-50k
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluation/logs/sit-fid-50k-%j.out
#SBATCH --error=evaluation/logs/sit-fid-50k-%j.err

set -euo pipefail

cd /home/guangyi.chen/workspace/jinyuan/SiT-Complementary
source /home/guangyi.chen/miniconda3/etc/profile.d/conda.sh
conda activate sit-complementary

export CUDA_VISIBLE_DEVICES=0,1,2,3

sample_root="evaluation/generated_50k"
reference="evaluation/VIRTUAL_imagenet256_labeled.npz"
mkdir -p "$sample_root" evaluation/logs evaluation/fid_results

run_evaluation() {
    local variant="$1"
    local checkpoint="$2"
    local cfg="$3"
    local folder="$sample_root/SiT-S-2-${variant}-4003200-cfg-${cfg}-64-ODE-250-dopri5"
    local archive="${folder}.npz"
    local result="evaluation/fid_results/${variant}-cfg-${cfg}.txt"

    echo "EVAL_BEGIN:${variant}:cfg${cfg}"
    if [[ ! -f "$archive" ]]; then
        torchrun --standalone --nnodes=1 --nproc_per_node=4 sample_ddp.py ODE \
            --variant "$variant" \
            --model SiT-S/2 \
            --num-fid-samples 50000 \
            --cfg-scale "$cfg" \
            --per-proc-batch-size 64 \
            --sample-dir "$sample_root" \
            --ckpt "$checkpoint"
    else
        echo "SAMPLE_ARCHIVE_EXISTS:${archive}"
    fi

    pytorch-fid "$folder" "$reference" \
        --device cuda:0 --batch-size 256 | tee "$result"
    echo "EVAL_PASS:${variant}:cfg${cfg}"
}

base_checkpoint="pretrained_models/BlueSourceJY-SiT-Complementary/checkpoints/bs256_lr1e-4/base/4003200.pt"
rotation_checkpoint="pretrained_models/BlueSourceJY-SiT-Complementary/checkpoints/bs256_lr1e-4/rotation-layer/4003200.pt"

run_evaluation base "$base_checkpoint" 1.0
run_evaluation base "$base_checkpoint" 4.0
run_evaluation rot-layer "$rotation_checkpoint" 1.0
run_evaluation rot-layer "$rotation_checkpoint" 4.0
