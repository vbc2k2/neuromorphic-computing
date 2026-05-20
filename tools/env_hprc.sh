#!/usr/bin/env bash
# Source this on HPRC before running the benchmark framework:
#
#   source tools/env_hprc.sh
#
# Assumes the local installs used during development:
#   /scratch/user/$USER/local/miniconda3
#   /scratch/user/$USER/local/verilator

SNN_SCRATCH="${SNN_SCRATCH:-/scratch/user/$USER}"
SNN_CONDA_ENV="${SNN_CONDA_ENV:-verilator-build}"

CONDA_SH="$SNN_SCRATCH/local/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$SNN_CONDA_ENV"
else
    echo "WARN: conda setup not found at $CONDA_SH" >&2
fi

VERILATOR_BIN="$SNN_SCRATCH/local/verilator/bin"
if [ -d "$VERILATOR_BIN" ]; then
    export PATH="$VERILATOR_BIN:$PATH"
else
    echo "WARN: Verilator bin not found at $VERILATOR_BIN" >&2
fi

export SNN_DATA_DIR="${SNN_DATA_DIR:-$SNN_SCRATCH/neuromorphic_data}"
mkdir -p "$SNN_DATA_DIR"

echo "python: $(command -v python3 2>/dev/null || true)"
python3 --version 2>/dev/null || true
echo "verilator: $(command -v verilator 2>/dev/null || true)"
verilator --version 2>/dev/null || true
