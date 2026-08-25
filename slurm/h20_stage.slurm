#!/usr/bin/env bash
#SBATCH --job-name=sit-h20-stage
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/h20-stage-%j.out
#SBATCH --error=slurm/h20-stage-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
: "${PIPELINE_VARIANT:?Missing PIPELINE_VARIANT}"
: "${PIPELINE_TARGET:?Missing PIPELINE_TARGET}"
bash workflow/run_h20_stage.sh "$PIPELINE_VARIANT" "$PIPELINE_TARGET"
