#!/usr/bin/env bash
#SBATCH --job-name=sit-fid-exact50k
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=evaluation/logs/sit-fid-exact50k-%j.out
#SBATCH --error=evaluation/logs/sit-fid-exact50k-%j.err

set -euo pipefail

cd /home/guangyi.chen/workspace/jinyuan/SiT-Complementary
source /home/guangyi.chen/miniconda3/etc/profile.d/conda.sh
conda activate sit-complementary

reference="evaluation/VIRTUAL_imagenet256_labeled.npz"

run_fid() {
    local variant="$1"
    local cfg="$2"
    local folder="evaluation/generated_50k/SiT-S-2-${variant}-4003200-cfg-${cfg}-64-ODE-250-dopri5"
    local result="evaluation/fid_results/${variant}-cfg-${cfg}.txt"

    echo "FID_BEGIN:${variant}:cfg${cfg}"
    pytorch-fid "$folder" "$reference" \
        --device cuda:0 --batch-size 256 | tee "$result"
    echo "FID_PASS:${variant}:cfg${cfg}"
}

run_fid base 1.0
run_fid base 4.0
run_fid rot-layer 1.0
run_fid rot-layer 4.0
