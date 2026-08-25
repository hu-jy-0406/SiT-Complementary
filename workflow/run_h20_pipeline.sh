#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PREPARE_ASSETS="${PREPARE_ASSETS:-auto}"
bash workflow/run_gpu_experiment.sh conv
export PREPARE_ASSETS=0
bash workflow/run_gpu_experiment.sh rotation-head

echo "GPU_PIPELINE_COMPLETE=$OUTPUT_ROOT/training_results/TRAINING_RESULTS.md"
