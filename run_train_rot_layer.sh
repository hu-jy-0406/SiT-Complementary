#!/usr/bin/env bash
set -euo pipefail

export RESULTS_DIR="${RESULTS_DIR:-results/rotation-layer}"
export RUN_NAME="${RUN_NAME:-SiT-S-2-RotationLayer-bs256-lr1e-4}"
export TRAINING_ENTRYPOINT=train_rot_layer.py

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_train.sh" "${@}"
