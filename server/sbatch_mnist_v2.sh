#!/usr/bin/env bash
# ============================================================================
# sbatch_mnist_v2.sh - SLURM array job for the v2 comparison:
#                      event SNN, dense SNN, and INT8 ANN baseline.
#
# Submit from the repository root:
#     sbatch server/sbatch_mnist_v2.sh
# ============================================================================
#SBATCH --job-name=snn-mnist-v2
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=11:59:00
#SBATCH --output=snn_mnist_v2_%A_%a.log

set -euo pipefail

if ! command -v xrun >/dev/null 2>&1; then
    module load xcelium >/dev/null 2>&1 || module load cadence >/dev/null 2>&1 || true
fi
if ! command -v xrun >/dev/null 2>&1; then
    echo "ERROR: xrun is not on PATH. Load the Cadence/Xcelium module for this cluster." >&2
    exit 1
fi

ROOT="$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && pwd)"
if [ ! -f "$ROOT/server/xcelium_run.sh" ] && [ -f "$ROOT/xcelium_run.sh" ]; then
    ROOT="$(cd "$ROOT/.." && pwd)"
fi
HERE="$ROOT/server"

NUM_TEST=300
NSLICE=6
STRIDE=$(( (NUM_TEST + NSLICE - 1) / NSLICE ))
K=${SLURM_ARRAY_TASK_ID:-0}
FIRST=$(( K * STRIDE ))
LAST=$(( FIRST + STRIDE - 1 ))
if [ "$LAST" -ge "$NUM_TEST" ]; then LAST=$(( NUM_TEST - 1 )); fi
TAG="_${K}"

echo "=== v2 slice $K : images $FIRST..$LAST ==="
bash "$HERE/xcelium_run.sh" event "$FIRST" "$LAST" "$TAG"
bash "$HERE/xcelium_run.sh" dense "$FIRST" "$LAST" "$TAG"
bash "$HERE/xcelium_run.sh" ann "$FIRST" "$LAST" "$TAG"
echo "=== v2 slice $K done ==="

