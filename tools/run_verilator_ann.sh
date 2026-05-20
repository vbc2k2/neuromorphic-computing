#!/usr/bin/env bash
# ============================================================================
# run_verilator_ann.sh - Open-source Verilator smoke/benchmark run for the
#                        INT8 ANN baseline.
#
# Usage:
#   bash tools/run_verilator_ann.sh [first] [last] [tag]
#
# Examples:
#   bash tools/run_verilator_ann.sh 0 4 _smoke
#   bash tools/run_verilator_ann.sh 0 49 _0
#
# Prerequisite:
#   python3 python/train_snn.py
#
# That command generates sim/snn_config.vh, sim/snn_labels.mem, and sim/ann_*.mem.
# ============================================================================
set -euo pipefail

FIRST="${1:-0}"
LAST="${2:-4}"
TAG="${3:-_vl}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SIM="$ROOT/sim"
RTL="$ROOT/rtl"
RUNDIR="$SIM/run_verilator_ann${TAG}"
OBJDIR="$RUNDIR/obj_dir"

command -v verilator >/dev/null 2>&1 || {
    echo "ERROR: verilator is not on PATH" >&2
    exit 1
}

mkdir -p "$RUNDIR"
rm -rf "$OBJDIR"
rm -f "$RUNDIR"/classify_ann"${TAG}".csv "$RUNDIR"/metrics_classify_ann"${TAG}".csv
rm -f "$SIM"/classify_ann"${TAG}".csv "$SIM"/metrics_classify_ann"${TAG}".csv

for f in snn_config.vh snn_labels.mem ann_pixels.mem ann_w1.mem ann_b1.mem ann_w2.mem ann_b2.mem; do
    if [ ! -f "$SIM/$f" ]; then
        echo "ERROR: missing $SIM/$f; run: python3 python/train_snn.py" >&2
        exit 1
    fi
    ln -sf "$SIM/$f" "$RUNDIR/$f"
done

echo "[verilator-ann$TAG] build  ($(date))"
cd "$ROOT"

cfg() {
    awk -v key="$1" '$2 == key { print $3 }' "$SIM/snn_config.vh"
}

N_INPUT="$(cfg SNN_N_INPUT)"
N_HIDDEN="$(cfg SNN_N_HIDDEN)"
N_OUTPUT="$(cfg SNN_N_OUTPUT)"
HIDDEN_SHIFT="$(cfg SNN_ANN_HIDDEN_SHIFT)"

for v in N_INPUT N_HIDDEN N_OUTPUT HIDDEN_SHIFT; do
    if [ -z "${!v}" ]; then
        echo "ERROR: $v missing from $SIM/snn_config.vh" >&2
        echo "Rerun python3 python/train_snn.py with the latest source." >&2
        exit 1
    fi
done

cp -f "$SIM/snn_config.vh" "$SIM/snn_config${TAG}.vh"

verilator -sv --cc \
    --Mdir "$OBJDIR" \
    -GN_INPUT="$N_INPUT" \
    -GN_HIDDEN="$N_HIDDEN" \
    -GN_OUTPUT="$N_OUTPUT" \
    -GHIDDEN_SHIFT="$HIDDEN_SHIFT" \
    "$RTL/top_ann.sv" \
    --top-module top_ann \
    --exe "$ROOT/tools/verilator_ann_main.cpp" \
    -CFLAGS "-std=c++17 -DSNN_N_INPUT=$N_INPUT -DSNN_N_OUTPUT=$N_OUTPUT"
make -C "$OBJDIR" -f Vtop_ann.mk -j "${VERILATOR_JOBS:-1}"

echo "[verilator-ann$TAG] images $FIRST..$LAST  ($(date))"
cd "$RUNDIR"
"$OBJDIR/Vtop_ann" \
    +first="$FIRST" +last="$LAST" +tag="$TAG"

cp -f "classify_ann${TAG}.csv" "$SIM/"
cp -f "metrics_classify_ann${TAG}.csv" "$SIM/"
echo "[verilator-ann$TAG] done  ($(date))"
