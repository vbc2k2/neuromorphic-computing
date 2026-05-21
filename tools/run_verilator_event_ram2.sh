#!/usr/bin/env bash
# ============================================================================
# run_verilator_event_ram2.sh - Verilator run for the 2-lane RAM-state SNN.
#
# Usage:
#   bash tools/run_verilator_event_ram2.sh [first] [last] [tag] [max_cycles_per_image]
# ============================================================================
set -euo pipefail

FIRST="${1:-0}"
LAST="${2:-4}"
TAG="${3:-_ram2}"
MAX_CYCLES="${4:-100000000}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SIM="$ROOT/sim"
RTL="$ROOT/rtl"
RUNDIR="$SIM/run_verilator_event_ram2${TAG}"
OBJDIR="$RUNDIR/obj_dir"

command -v verilator >/dev/null 2>&1 || {
    echo "ERROR: verilator is not on PATH" >&2
    exit 1
}

mkdir -p "$RUNDIR"
rm -rf "$OBJDIR"
rm -f "$RUNDIR"/classify_event"${TAG}".csv "$RUNDIR"/metrics_classify_event"${TAG}".csv
rm -f "$SIM"/classify_event"${TAG}".csv "$SIM"/metrics_classify_event"${TAG}".csv

for f in snn_config.vh snn_labels.mem snn_spikes.mem csr_dst.mem csr_weight.mem csr_start.mem csr_count.mem; do
    if [ ! -f "$SIM/$f" ]; then
        echo "ERROR: missing $SIM/$f; run an exporter first" >&2
        exit 1
    fi
    ln -sf "$SIM/$f" "$RUNDIR/$f"
done

cfg() {
    awk -v key="$1" '$2 == key { print $3 }' "$SIM/snn_config.vh"
}

N_TOTAL="$(cfg SNN_N_TOTAL)"
N_INPUT="$(cfg SNN_N_INPUT)"
N_OUTPUT="$(cfg SNN_N_OUTPUT)"
ID_BIAS="$(cfg SNN_ID_BIAS)"
OUT_BASE="$(cfg SNN_ID_OUTPUT_BASE)"
T_STEPS="$(cfg SNN_T_STEPS)"
THRESHOLD="$(cfg SNN_THRESHOLD)"
LEAK="$(cfg SNN_LEAK)"
NUM_SYN="$(cfg SNN_NUM_SYN)"

for v in N_TOTAL N_INPUT N_OUTPUT ID_BIAS OUT_BASE T_STEPS THRESHOLD LEAK NUM_SYN; do
    if [ -z "${!v}" ]; then
        echo "ERROR: $v missing from $SIM/snn_config.vh" >&2
        exit 1
    fi
done

cp -f "$SIM/snn_config.vh" "$SIM/snn_config${TAG}.vh"

echo "[verilator-event-ram2$TAG] build  ($(date))"
cd "$ROOT"

verilator -sv --cc \
    --Mdir "$OBJDIR" \
    -GNUM_NEURONS="$N_TOTAL" \
    -GTHRESHOLD="$THRESHOLD" \
    -GLEAK="$LEAK" \
    -GNUM_SYN="$NUM_SYN" \
    -GFIFO_DEPTH=4096 \
    "$RTL/synapse_csr.sv" \
    "$RTL/top_event_ram2.sv" \
    --top-module top_event_ram2 \
    --exe "$ROOT/tools/verilator_event_main.cpp" \
    -CFLAGS "-std=c++17 -DEVENT_TOP_HEADER=\\\"Vtop_event_ram2.h\\\" -DEVENT_TOP_CLASS=Vtop_event_ram2 -DEVENT_DESIGN_NAME=\\\"event_ram2_verilator\\\" -DSNN_N_TOTAL=$N_TOTAL -DSNN_N_INPUT=$N_INPUT -DSNN_N_OUTPUT=$N_OUTPUT -DSNN_ID_BIAS=$ID_BIAS -DSNN_ID_OUTPUT_BASE=$OUT_BASE -DSNN_T_STEPS=$T_STEPS"
make -C "$OBJDIR" -f Vtop_event_ram2.mk -j "${VERILATOR_JOBS:-1}"

echo "[verilator-event-ram2$TAG] images $FIRST..$LAST  ($(date))"
cd "$RUNDIR"
"$OBJDIR/Vtop_event_ram2" \
    +first="$FIRST" +last="$LAST" +tag="$TAG" +max_cycles="$MAX_CYCLES"

cp -f "classify_event${TAG}.csv" "$SIM/"
cp -f "metrics_classify_event${TAG}.csv" "$SIM/"
echo "[verilator-event-ram2$TAG] done  ($(date))"
