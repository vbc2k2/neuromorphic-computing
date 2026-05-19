#!/usr/bin/env bash
# ============================================================================
# sbatch_mnist.sh - SLURM array job: run the MNIST SNN benchmark across cores.
#
#   The test set (SNN_NUM_TEST images, default 300) is split into 6 slices;
#   each array task runs one slice through BOTH the event-driven design and
#   the dense baseline on Xcelium. Merge the per-slice results afterwards with
#       python python/merge_results.py
#
# Submit:  sbatch server/sbatch_mnist.sh
# Adjust the SBATCH directives / module name to your cluster.
# ============================================================================
#SBATCH --job-name=snn-mnist
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=snn_mnist_%A_%a.log

set -euo pipefail

# --- cluster-specific: make `xrun` available -------------------------------
if ! command -v xrun >/dev/null 2>&1; then
    module load xcelium >/dev/null 2>&1 || module load cadence >/dev/null 2>&1 || true
fi
if ! command -v xrun >/dev/null 2>&1; then
    echo "ERROR: xrun is not on PATH. Load the Cadence/Xcelium module for this cluster." >&2
    exit 1
fi

# SLURM may execute this script from a spool copy, so do not derive paths from
# BASH_SOURCE. Anchor at the directory where sbatch was submitted.
ROOT="$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && pwd)"
if [ ! -f "$ROOT/server/xcelium_run.sh" ] && [ -f "$ROOT/xcelium_run.sh" ]; then
    ROOT="$(cd "$ROOT/.." && pwd)"
fi
HERE="$ROOT/server"

# --- image slice for this array task ---------------------------------------
NUM_TEST=300                       # must match SNN_NUM_TEST in sim/snn_config.vh
NSLICE=6                           # = array size
STRIDE=$(( (NUM_TEST + NSLICE - 1) / NSLICE ))
K=${SLURM_ARRAY_TASK_ID:-0}
FIRST=$(( K * STRIDE ))
LAST=$(( FIRST + STRIDE - 1 ))
if [ "$LAST" -ge "$NUM_TEST" ]; then LAST=$(( NUM_TEST - 1 )); fi
TAG="_${K}"

echo "=== slice $K : images $FIRST..$LAST ==="
bash "$HERE/xcelium_run.sh" event "$FIRST" "$LAST" "$TAG"
bash "$HERE/xcelium_run.sh" dense "$FIRST" "$LAST" "$TAG"
echo "=== slice $K done ==="
