#!/usr/bin/env bash
# ============================================================================
# xcelium_run.sh - Compile and run one SNN accelerator design on Cadence
#                  Xcelium, for a given range of test images.
#
# Usage:   xcelium_run.sh <event|dense|ann> <first> <last> <tag>
# Example: xcelium_run.sh event 0 49 _0      # images 0..49, output suffix _0
#
# Each invocation runs in its own work directory (run_<design><tag>/) so many
# slices can run concurrently under sbatch without clobbering each other.
# Result CSVs are copied back into sim/.
# ============================================================================
set -euo pipefail

DESIGN="${1:?need event|dense|ann}"
FIRST="${2:?need first image}"
LAST="${3:?need last image}"
TAG="${4:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SIM="$ROOT/sim"; RTL="$ROOT/rtl"; TB="$ROOT/tb"

if [ "$DESIGN" = event ]; then
    SRCS=("$RTL/neuron_core.sv" "$RTL/synapse_csr.sv" "$RTL/spike_router.sv" \
          "$RTL/top.sv" "$TB/tb_classify.sv")
    TOP=tb_classify
elif [ "$DESIGN" = dense ]; then
    SRCS=("$RTL/neuron_core.sv" "$RTL/synapse_csr.sv" "$RTL/top_dense.sv" \
          "$TB/tb_classify_dense.sv")
    TOP=tb_classify_dense
elif [ "$DESIGN" = ann ]; then
    SRCS=("$RTL/top_ann.sv" "$TB/tb_classify_ann.sv")
    TOP=tb_classify_ann
else
    echo "design must be 'event', 'dense', or 'ann'" >&2; exit 1
fi

RUNDIR="$SIM/run_${DESIGN}${TAG}"
mkdir -p "$RUNDIR"; cd "$RUNDIR"

LOG="xrun_${DESIGN}${TAG}.log"
OUT_CSV="classify_${DESIGN}${TAG}.csv"
METRICS_CSV="metrics_classify_${DESIGN}${TAG}.csv"
TIMEOUT_NS="${SNN_TIMEOUT_NS:-120000000000}"
rm -f "$LOG" "$OUT_CSV" "$METRICS_CSV" \
      "$SIM/$OUT_CSV" "$SIM/$METRICS_CSV"

# The testbench opens these by relative name - link them into the run dir.
MEMS=(snn_config.vh snn_labels.mem)
if [ "$DESIGN" = ann ]; then
    MEMS+=(ann_pixels.mem ann_w1.mem ann_b1.mem ann_w2.mem ann_b2.mem)
else
    MEMS+=(snn_spikes.mem csr_dst.mem csr_weight.mem csr_start.mem csr_count.mem)
fi
for f in "${MEMS[@]}"; do
    if [ ! -f "$SIM/$f" ]; then
        echo "[$DESIGN$TAG] missing $SIM/$f; rerun python/train_snn.py" >&2
        exit 1
    fi
    ln -sf "$SIM/$f" .
done

echo "[$DESIGN$TAG] images $FIRST..$LAST  ($(date))"
xrun -sv -timescale 1ns/1ps -incdir . "${SRCS[@]}" -top "$TOP" \
     +first="$FIRST" +last="$LAST" +tag="$TAG" +timeout_ns="$TIMEOUT_NS" \
     -l "$LOG"

if grep -Eq '^\$readmem error:|\[TIMEOUT\]|xrun: \*E,|xmvlog: \*E,|xmelab: \*E,|xmsim: \*E,' "$LOG"; then
    echo "[$DESIGN$TAG] Xcelium reported errors; see $RUNDIR/$LOG" >&2
    exit 1
fi

if [ ! -s "$OUT_CSV" ] || [ ! -s "$METRICS_CSV" ]; then
    echo "[$DESIGN$TAG] missing expected CSV output files in $RUNDIR" >&2
    exit 1
fi

cp -f "$OUT_CSV" "$SIM/"
cp -f "$METRICS_CSV" "$SIM/"
echo "[$DESIGN$TAG] done  ($(date))"
