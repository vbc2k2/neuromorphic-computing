#!/usr/bin/env bash
# ============================================================================
# run_nmnist_saliency_validate.sh - Resumable validation for N-MNIST saliency
#                                  membrane-readout candidates.
#
# Run from repo root after:
#   source tools/env_hprc.sh
#
# The script is intentionally resumable. If a metric/log already exists for a
# tag, that stage is skipped. Set FORCE=1 to rerun everything.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SIM="$ROOT/sim"
RESULTS="$ROOT/results"
LOGDIR="$RESULTS/nmnist_saliency_validate/logs"
MAX_CYCLES="${MAX_CYCLES:-400000000}"
FORCE="${FORCE:-0}"
RUN_SYNTH="${RUN_SYNTH:-1}"
FORCE_SYNTH="${FORCE_SYNTH:-0}"

mkdir -p "$LOGDIR"
cd "$ROOT"

need_tool() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: $1 is not on PATH. On HPRC run: source tools/env_hprc.sh" >&2
        exit 1
    }
}

metric_exists() {
    local kind="$1"
    local tag="$2"
    [ "$FORCE" != "1" ] && [ -s "$SIM/metrics_classify_${kind}${tag}.csv" ]
}

log_exists() {
    local path="$1"
    [ "$FORCE" != "1" ] && [ "$FORCE_SYNTH" != "1" ] && [ -s "$path" ]
}

run_logged() {
    local name="$1"
    shift
    local logfile="$LOGDIR/${name}.log"
    echo "+ $*"
    "$@" 2>&1 | tee "$logfile"
}

archive_tag() {
    local tag="$1"
    local out="$RESULTS/report${tag}"
    mkdir -p "$out"

    cp -f "$SIM/metrics_classify_event${tag}.csv" "$out/" 2>/dev/null || true
    cp -f "$SIM/metrics_classify_ann${tag}.csv" "$out/" 2>/dev/null || true
    cp -f "$SIM/classify_event${tag}.csv" "$out/" 2>/dev/null || true
    cp -f "$SIM/snn_golden_predictions${tag}.csv" "$out/" 2>/dev/null || true
    cp -f "$SIM/snn_config${tag}.vh" "$out/" 2>/dev/null || true
    cp -f "$RESULTS"/yosys_*"${tag}"_*.log "$out/" 2>/dev/null || true
    cp -f "$LOGDIR"/*"${tag}"*.log "$out/" 2>/dev/null || true

    if [ -s "$SIM/metrics_classify_event${tag}.csv" ] && [ -s "$SIM/metrics_classify_ann${tag}.csv" ]; then
        python3 tools/report.py --tag "$tag" --event-design event_ram2 \
            | tee "$out/report.txt"
        python3 tools/report.py --tag "$tag" --event-design event_ram2 --format md \
            > "$out/report.md"
    fi
}

validate_candidate() {
    local tag="$1"
    local t_steps="$2"
    local topk="$3"
    local bias="$4"
    local prune="$5"

    echo
    echo "=============================================================================="
    echo "Candidate $tag"
    echo "  T=$t_steps TOPK=$topk BIAS=$bias PRUNE=$prune READOUT=membrane"
    echo "=============================================================================="

    local needs_rtl_inputs=0
    if ! metric_exists event "$tag" || ! metric_exists ann "$tag"; then
        needs_rtl_inputs=1
    fi

    if [ "$FORCE" = "1" ] || [ "$needs_rtl_inputs" = "1" ] || [ ! -s "$SIM/snn_config${tag}.vh" ] || [ ! -s "$SIM/snn_golden_predictions${tag}.csv" ]; then
        run_logged "export${tag}" \
            python3 tools/bench.py export nmnist \
                --set "NMNIST_T_STEPS=$t_steps" \
                --set "NMNIST_TOPK_W1=$topk" \
                --set "NMNIST_FINETUNE_EPOCHS=12" \
                --set "NMNIST_BIAS_MODE=$bias" \
                --set "NMNIST_READOUT=membrane" \
                --set "NMNIST_PRUNE_MODE=$prune"
        cp -f "$SIM/snn_golden_predictions.csv" "$SIM/snn_golden_predictions${tag}.csv"
    else
        echo "[skip] export not needed for completed RTL metrics: $SIM/snn_config${tag}.vh"
    fi

    if ! metric_exists event "$tag"; then
        run_logged "event${tag}" \
            python3 tools/bench.py run nmnist --design event_ram2 \
                --tag "$tag" --max-cycles "$MAX_CYCLES"
    else
        echo "[skip] event metrics exist: $SIM/metrics_classify_event${tag}.csv"
    fi

    if ! metric_exists ann "$tag"; then
        run_logged "ann${tag}" \
            python3 tools/bench.py run nmnist --design ann --tag "$tag"
    else
        echo "[skip] ANN metrics exist: $SIM/metrics_classify_ann${tag}.csv"
    fi

    if [ -s "$SIM/snn_golden_predictions${tag}.csv" ] && [ -s "$SIM/classify_event${tag}.csv" ]; then
        run_logged "compare${tag}" \
            python3 tools/compare_predictions.py --tag "$tag" \
                --golden "$SIM/snn_golden_predictions${tag}.csv"
    else
        echo "[skip] prediction compare missing golden or RTL classify CSV"
    fi

    run_logged "summary${tag}" \
        python3 tools/bench.py summarize nmnist --tag "$tag"

    if [ "$RUN_SYNTH" = "1" ]; then
        for design in event_ram2 ann; do
            for flow in xilinx asic; do
                local kind="$design"
                local log="$RESULTS/yosys_${kind}${tag}_${flow}.log"
                if ! log_exists "$log"; then
                    run_logged "yosys_${design}_${flow}${tag}" \
                        bash tools/run_yosys_synth.sh "$tag" "$design" "$flow"
                else
                    echo "[skip] Yosys log exists: $log"
                fi
            done
        done
    else
        echo "[skip] synthesis disabled: RUN_SYNTH=$RUN_SYNTH"
    fi

    archive_tag "$tag"
}

need_tool python3
need_tool bash
need_tool verilator
if [ "$RUN_SYNTH" = "1" ]; then
    need_tool yosys
fi

echo "repo: $ROOT"
echo "logs: $LOGDIR"
echo "FORCE=$FORCE FORCE_SYNTH=$FORCE_SYNTH RUN_SYNTH=$RUN_SYNTH MAX_CYCLES=$MAX_CYCLES"

validate_candidate "_nmnist_t50_k128_saliency_mem" 50 128 none saliency
validate_candidate "_nmnist_t50_k96_saliency_mem" 50 96 none saliency

echo
echo "=============================================================================="
echo "Final reports"
echo "=============================================================================="
python3 tools/report.py --tag _nmnist_t50_k128_saliency_mem --event-design event_ram2 || true
python3 tools/report.py --tag _nmnist_t50_k96_saliency_mem --event-design event_ram2 || true
