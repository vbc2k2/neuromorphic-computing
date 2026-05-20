#!/usr/bin/env bash
# Run the lint checks that should pass with open-source Verilator.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

verilator --lint-only -sv rtl/neuron_core.sv rtl/synapse_csr.sv rtl/spike_router.sv rtl/top.sv
verilator --lint-only -sv rtl/neuron_core.sv rtl/synapse_csr.sv rtl/top_dense.sv
verilator --lint-only -sv rtl/top_ann.sv

if [ -f sim/snn_config.vh ]; then
    verilator --lint-only -sv --timing -Isim rtl/neuron_core.sv rtl/synapse_csr.sv rtl/spike_router.sv rtl/top.sv tb/tb_classify.sv
    verilator --lint-only -sv --timing -Isim rtl/neuron_core.sv rtl/synapse_csr.sv rtl/top_dense.sv tb/tb_classify_dense.sv
    verilator --lint-only -sv --timing -Isim rtl/top_ann.sv tb/tb_classify_ann.sv
else
    echo "Skipping testbench lint because sim/snn_config.vh is missing."
    echo "Run python3 python/train_snn.py, then rerun this script."
fi

