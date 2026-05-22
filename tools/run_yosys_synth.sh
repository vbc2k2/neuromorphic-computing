#!/usr/bin/env bash
# ============================================================================
# run_yosys_synth.sh - Open-source synthesis/stat flow for apples-to-apples
#                      event SNN vs INT8 ANN comparisons.
#
# Usage:
#   bash tools/run_yosys_synth.sh [tag] [design] [flow]
#
# Examples:
#   bash tools/run_yosys_synth.sh _nmnist_pipe all generic
#   bash tools/run_yosys_synth.sh _nmnist_pipe event xilinx
#   bash tools/run_yosys_synth.sh _nmnist_pipe event_ram xilinx
#   bash tools/run_yosys_synth.sh _nmnist_pipe event_ram2 xilinx
#   ASIC_LIBERTY=/path/to/sky130.lib bash tools/run_yosys_synth.sh _nmnist_pipe event_ram asic
#
# Arguments:
#   tag     matches the Verilator/bench tag. Uses sim/snn_config${tag}.vh when
#           present, otherwise sim/snn_config.vh.
#   design  event, event_ram, event_ram2, ann, or all.
#   flow    generic: technology-independent memory-aware stats.
#           xilinx:  synth_xilinx LUT/FF/BRAM/DSP estimate, excluding I/O pads.
#           asic:    ASIC-oriented logic mapping. Keeps memories abstract and
#                    uses ASIC_LIBERTY or LIBERTY_FILE for standard-cell area
#                    when provided.
# ============================================================================
set -euo pipefail

TAG="${1:-_vl}"
DESIGN="${2:-all}"
FLOW="${3:-generic}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SIM="$ROOT/sim"
RTL="$ROOT/rtl"
RESULTS="$ROOT/results"
CFG="$SIM/snn_config${TAG}.vh"
ASIC_LIBERTY_FILE="${ASIC_LIBERTY:-${LIBERTY_FILE:-}}"

command -v yosys >/dev/null 2>&1 || {
    echo "ERROR: yosys is not on PATH" >&2
    exit 1
}

if [ ! -f "$CFG" ]; then
    CFG="$SIM/snn_config.vh"
fi
if [ ! -f "$CFG" ]; then
    echo "ERROR: missing sim/snn_config.vh; export a benchmark first" >&2
    exit 1
fi

mkdir -p "$RESULTS"

cfg() {
    awk -v key="$1" '$2 == key { gsub(/"/, "", $3); print $3 }' "$CFG"
}

N_TOTAL="$(cfg SNN_N_TOTAL)"
N_INPUT="$(cfg SNN_N_INPUT)"
N_HIDDEN="$(cfg SNN_N_HIDDEN)"
N_OUTPUT="$(cfg SNN_N_OUTPUT)"
OUT_BASE="$(cfg SNN_ID_OUTPUT_BASE)"
THRESHOLD="$(cfg SNN_THRESHOLD)"
LEAK="$(cfg SNN_LEAK)"
NUM_SYN="$(cfg SNN_NUM_SYN)"
HIDDEN_SHIFT="$(cfg SNN_ANN_HIDDEN_SHIFT)"

require_cfg() {
    local name="$1"
    if [ -z "${!name}" ]; then
        echo "ERROR: $name missing from $CFG" >&2
        exit 1
    fi
}

for v in N_INPUT N_HIDDEN N_OUTPUT; do
    require_cfg "$v"
done

asic_steps() {
    if [ -n "$ASIC_LIBERTY_FILE" ]; then
        cat <<EOF
                    proc; opt; memory -nomap; opt
                    techmap; opt
                    dfflibmap -liberty $ASIC_LIBERTY_FILE
                    abc -liberty $ASIC_LIBERTY_FILE
                    clean
                    stat -liberty $ASIC_LIBERTY_FILE
EOF
    else
        cat <<EOF
                    proc; opt; memory -nomap; opt
                    techmap; opt
                    stat
EOF
    fi
}

print_asic_note() {
    if [ "$FLOW" = "asic" ]; then
        if [ -n "$ASIC_LIBERTY_FILE" ]; then
            echo "  ASIC Liberty: $ASIC_LIBERTY_FILE"
        else
            echo "  ASIC Liberty: not set; reporting generic mapped logic and abstract memories"
            echo "  Set ASIC_LIBERTY=/path/to/*.lib for standard-cell area units."
        fi
    fi
}

run_event() {
    for v in N_TOTAL THRESHOLD LEAK NUM_SYN; do
        require_cfg "$v"
    done

    local log="$RESULTS/yosys_event${TAG}_${FLOW}.log"
    echo "[yosys-event$TAG] flow=$FLOW config=$(basename "$CFG")"
    print_asic_note
    (
        cd "$SIM"
        case "$FLOW" in
            generic)
                yosys -l "$log" -p "
                    read_verilog -sv -defer -DYOSYS ../rtl/neuron_core.sv ../rtl/synapse_csr.sv ../rtl/spike_router.sv ../rtl/top.sv
                    hierarchy -top top \
                        -chparam NUM_NEURONS $N_TOTAL \
                        -chparam THRESHOLD $THRESHOLD \
                        -chparam LEAK $LEAK \
                        -chparam NUM_SYN $NUM_SYN \
                        -chparam FIFO_DEPTH 2048
                    proc; opt; memory -nomap; opt; stat
                "
                ;;
            xilinx)
                yosys -l "$log" -p "
                    read_verilog -sv -defer -DYOSYS ../rtl/neuron_core.sv ../rtl/synapse_csr.sv ../rtl/spike_router.sv ../rtl/top.sv
                    hierarchy -top top \
                        -chparam NUM_NEURONS $N_TOTAL \
                        -chparam THRESHOLD $THRESHOLD \
                        -chparam LEAK $LEAK \
                        -chparam NUM_SYN $NUM_SYN \
                        -chparam FIFO_DEPTH 2048
                    synth_xilinx -family xc7 -noiopad
                    stat
                "
                ;;
            asic)
                yosys -l "$log" -p "
                    read_verilog -sv -defer -DYOSYS ../rtl/neuron_core.sv ../rtl/synapse_csr.sv ../rtl/spike_router.sv ../rtl/top.sv
                    hierarchy -top top \
                        -chparam NUM_NEURONS $N_TOTAL \
                        -chparam THRESHOLD $THRESHOLD \
                        -chparam LEAK $LEAK \
                        -chparam NUM_SYN $NUM_SYN \
                        -chparam FIFO_DEPTH 2048
$(asic_steps)
                "
                ;;
            *)
                echo "ERROR: unknown flow '$FLOW' (use generic, xilinx, or asic)" >&2
                exit 1
                ;;
        esac
    )
    echo "[yosys-event$TAG] wrote $log"
}

run_ann() {
    require_cfg HIDDEN_SHIFT

    local log="$RESULTS/yosys_ann${TAG}_${FLOW}.log"
    echo "[yosys-ann$TAG] flow=$FLOW config=$(basename "$CFG")"
    print_asic_note
    (
        cd "$SIM"
        case "$FLOW" in
            generic)
                yosys -l "$log" -p "
                    read_verilog -sv -defer ../rtl/top_ann.sv
                    hierarchy -top top_ann \
                        -chparam N_INPUT $N_INPUT \
                        -chparam N_HIDDEN $N_HIDDEN \
                        -chparam N_OUTPUT $N_OUTPUT \
                        -chparam HIDDEN_SHIFT $HIDDEN_SHIFT
                    proc; opt; memory -nomap; opt; stat
                "
                ;;
            xilinx)
                yosys -l "$log" -p "
                    read_verilog -sv -defer ../rtl/top_ann.sv
                    hierarchy -top top_ann \
                        -chparam N_INPUT $N_INPUT \
                        -chparam N_HIDDEN $N_HIDDEN \
                        -chparam N_OUTPUT $N_OUTPUT \
                        -chparam HIDDEN_SHIFT $HIDDEN_SHIFT
                    synth_xilinx -family xc7 -noiopad
                    stat
                "
                ;;
            asic)
                yosys -l "$log" -p "
                    read_verilog -sv -defer ../rtl/top_ann.sv
                    hierarchy -top top_ann \
                        -chparam N_INPUT $N_INPUT \
                        -chparam N_HIDDEN $N_HIDDEN \
                        -chparam N_OUTPUT $N_OUTPUT \
                        -chparam HIDDEN_SHIFT $HIDDEN_SHIFT
$(asic_steps)
                "
                ;;
            *)
                echo "ERROR: unknown flow '$FLOW' (use generic, xilinx, or asic)" >&2
                exit 1
                ;;
        esac
    )
    echo "[yosys-ann$TAG] wrote $log"
}

run_event_ram() {
    for v in N_TOTAL THRESHOLD LEAK NUM_SYN; do
        require_cfg "$v"
    done

    local log="$RESULTS/yosys_event_ram${TAG}_${FLOW}.log"
    echo "[yosys-event-ram$TAG] flow=$FLOW config=$(basename "$CFG")"
    print_asic_note
    (
        cd "$SIM"
        case "$FLOW" in
            generic)
                yosys -l "$log" -p "
                    read_verilog -sv -defer -DYOSYS ../rtl/synapse_csr.sv ../rtl/top_event_ram.sv
                    hierarchy -top top_event_ram \
                        -chparam NUM_NEURONS $N_TOTAL \
                        -chparam THRESHOLD $THRESHOLD \
                        -chparam LEAK $LEAK \
                        -chparam NUM_SYN $NUM_SYN \
                        -chparam FIFO_DEPTH 4096
                    proc; opt; memory -nomap; opt; stat
                "
                ;;
            xilinx)
                yosys -l "$log" -p "
                    read_verilog -sv -defer -DYOSYS ../rtl/synapse_csr.sv ../rtl/top_event_ram.sv
                    hierarchy -top top_event_ram \
                        -chparam NUM_NEURONS $N_TOTAL \
                        -chparam THRESHOLD $THRESHOLD \
                        -chparam LEAK $LEAK \
                        -chparam NUM_SYN $NUM_SYN \
                        -chparam FIFO_DEPTH 4096
                    synth_xilinx -family xc7 -noiopad
                    stat
                "
                ;;
            asic)
                yosys -l "$log" -p "
                    read_verilog -sv -defer -DYOSYS ../rtl/synapse_csr.sv ../rtl/top_event_ram.sv
                    hierarchy -top top_event_ram \
                        -chparam NUM_NEURONS $N_TOTAL \
                        -chparam THRESHOLD $THRESHOLD \
                        -chparam LEAK $LEAK \
                        -chparam NUM_SYN $NUM_SYN \
                        -chparam FIFO_DEPTH 4096
$(asic_steps)
                "
                ;;
            *)
                echo "ERROR: unknown flow '$FLOW' (use generic, xilinx, or asic)" >&2
                exit 1
                ;;
        esac
    )
    echo "[yosys-event-ram$TAG] wrote $log"
}

run_event_ram2() {
    for v in N_TOTAL N_OUTPUT OUT_BASE THRESHOLD LEAK NUM_SYN; do
        require_cfg "$v"
    done

    local log="$RESULTS/yosys_event_ram2${TAG}_${FLOW}.log"
    echo "[yosys-event-ram2$TAG] flow=$FLOW config=$(basename "$CFG")"
    print_asic_note
    (
        cd "$SIM"
        case "$FLOW" in
            generic)
                yosys -l "$log" -p "
                    read_verilog -sv -defer -DYOSYS ../rtl/synapse_csr.sv ../rtl/top_event_ram2.sv
                    hierarchy -top top_event_ram2 \
                        -chparam NUM_NEURONS $N_TOTAL \
                        -chparam NUM_OUTPUT $N_OUTPUT \
                        -chparam ID_OUTPUT_BASE $OUT_BASE \
                        -chparam THRESHOLD $THRESHOLD \
                        -chparam LEAK $LEAK \
                        -chparam NUM_SYN $NUM_SYN \
                        -chparam FIFO_DEPTH 4096
                    proc; opt; memory -nomap; opt; stat
                "
                ;;
            xilinx)
                yosys -l "$log" -p "
                    read_verilog -sv -defer -DYOSYS ../rtl/synapse_csr.sv ../rtl/top_event_ram2.sv
                    hierarchy -top top_event_ram2 \
                        -chparam NUM_NEURONS $N_TOTAL \
                        -chparam NUM_OUTPUT $N_OUTPUT \
                        -chparam ID_OUTPUT_BASE $OUT_BASE \
                        -chparam THRESHOLD $THRESHOLD \
                        -chparam LEAK $LEAK \
                        -chparam NUM_SYN $NUM_SYN \
                        -chparam FIFO_DEPTH 4096
                    synth_xilinx -family xc7 -noiopad
                    stat
                "
                ;;
            asic)
                yosys -l "$log" -p "
                    read_verilog -sv -defer -DYOSYS ../rtl/synapse_csr.sv ../rtl/top_event_ram2.sv
                    hierarchy -top top_event_ram2 \
                        -chparam NUM_NEURONS $N_TOTAL \
                        -chparam NUM_OUTPUT $N_OUTPUT \
                        -chparam ID_OUTPUT_BASE $OUT_BASE \
                        -chparam THRESHOLD $THRESHOLD \
                        -chparam LEAK $LEAK \
                        -chparam NUM_SYN $NUM_SYN \
                        -chparam FIFO_DEPTH 4096
$(asic_steps)
                "
                ;;
            *)
                echo "ERROR: unknown flow '$FLOW' (use generic, xilinx, or asic)" >&2
                exit 1
                ;;
        esac
    )
    echo "[yosys-event-ram2$TAG] wrote $log"
}

case "$DESIGN" in
    event)     run_event ;;
    event_ram) run_event_ram ;;
    event_ram2) run_event_ram2 ;;
    ann)       run_ann ;;
    all)       run_event; run_ann ;;
    *)
        echo "ERROR: unknown design '$DESIGN' (use event, event_ram, event_ram2, ann, or all)" >&2
        exit 1
        ;;
esac
