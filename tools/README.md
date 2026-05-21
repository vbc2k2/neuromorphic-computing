# Benchmark Tools

Use `tools/bench.py` as the main entry point for exports, Verilator runs, and
summary tables.

On HPRC:

```bash
source tools/env_hprc.sh
python3 tools/bench.py doctor
```

List benchmarks:

```bash
python3 tools/bench.py list
```

Run a complete benchmark:

```bash
python3 tools/bench.py all sparse
python3 tools/bench.py all nmnist
```

Run step by step:

```bash
python3 tools/bench.py export nmnist
python3 tools/bench.py run nmnist --design event --last 49
python3 tools/bench.py run nmnist --design ann --last 49
python3 tools/bench.py summarize nmnist
```

Run the area-oriented RAM-state event prototype:

```bash
python3 tools/bench.py run nmnist --design event_ram --tag _nmnist_t50_k128_ram --max-cycles 200000000
python3 tools/bench.py summarize nmnist --tag _nmnist_t50_k128_ram
```

Run the 2-lane RAM-state event prototype:

```bash
python3 tools/bench.py run nmnist --design event_ram2 --tag _nmnist_t50_k128_ram2lane --first 0 --last 9 --max-cycles 300000000
python3 tools/bench.py run nmnist --design event_ram2 --tag _nmnist_t50_k128_ram2lane --max-cycles 300000000
python3 tools/bench.py run nmnist --design ann --tag _nmnist_t50_k128_ram2lane
bash tools/run_yosys_synth.sh _nmnist_t50_k128_ram2lane event_ram2 xilinx
bash tools/run_yosys_synth.sh _nmnist_t50_k128_ram2lane event_ram2 asic
python3 tools/report.py --tag _nmnist_t50_k128_ram2lane --event-design event_ram2
```

Run open-source synthesis/stat estimates after a tagged RTL run:

```bash
bash tools/run_yosys_synth.sh _nmnist_pipe all generic
bash tools/run_yosys_synth.sh _nmnist_pipe all xilinx
bash tools/run_yosys_synth.sh _nmnist_pipe event_ram xilinx
```

Run ASIC-oriented logic stats. Without a Liberty file this reports generic
mapped logic and abstract memories; with `ASIC_LIBERTY` it also reports
standard-cell area units from that library.

```bash
bash tools/run_yosys_synth.sh _nmnist_t50_k128_bram event_ram asic
bash tools/run_yosys_synth.sh _nmnist_t50_k128_bram ann asic

ASIC_LIBERTY=/path/to/sky130_fd_sc_hd__tt_025C_1v80.lib \
  bash tools/run_yosys_synth.sh _nmnist_t50_k128_bram event_ram asic
ASIC_LIBERTY=/path/to/sky130_fd_sc_hd__tt_025C_1v80.lib \
  bash tools/run_yosys_synth.sh _nmnist_t50_k128_bram ann asic
```

The ASIC mode is not a routed/post-layout result. It keeps SRAM-like memories
abstract, so compare logic area separately from model/state memory bits.

Generate one combined report for a tag:

```bash
python3 tools/report.py --tag _nmnist_t50_k128_bram --event-design event_ram
python3 tools/report.py --tag _nmnist_t50_k128_bram --event-design event_ram --format md
```

The Yosys wrapper uses the frozen `sim/snn_config<tag>.vh` snapshot when it
exists, runs from `sim/` so `$readmemh` files resolve correctly, and applies the
same parameters used by the Verilator benchmark.

Pass exporter tuning parameters:

```bash
python3 tools/bench.py export nmnist --set NMNIST_EPOCHS=25 --set NMNIST_TOPK_W1=96
python3 tools/bench.py export shd --set SHD_N_HIDDEN=128 --set SHD_EPOCHS=50
```

N-MNIST uses deterministic balanced train/test subsets by default. Useful
accuracy/area sweeps:

```bash
python3 tools/bench.py export nmnist --set NMNIST_TOPK_W1=128 --set NMNIST_FINETUNE_EPOCHS=12
python3 tools/bench.py export nmnist --set NMNIST_TOPK_W1=256 --set NMNIST_FINETUNE_EPOCHS=20
```

For algorithmic tuning before spending time on RTL/synthesis, run the
export-only tuner. It trains/exports each point, reads
`sim/export_nmnist_metrics.json`, and ranks candidates by accuracy, synapses,
and estimated synapse deliveries:

```bash
# Small sanity sweep.
python3 tools/tune_nmnist.py --preset quick --snapshot

# Broader edge-efficiency sweep.
python3 tools/tune_nmnist.py --preset edge --snapshot --out results/nmnist_tune_edge/summary.csv

# Override or add exact knobs.
python3 tools/tune_nmnist.py \
  --grid NMNIST_T_STEPS=40,50 \
  --grid NMNIST_TOPK_W1=96,128,160 \
  --grid NMNIST_BIAS_MODE=none,hidden \
  --grid NMNIST_ACTIVITY_LAMBDA=0,0.0001
```

Useful N-MNIST exporter knobs:

- `NMNIST_BIAS_MODE=none|hidden|all`: use the already-injected bias neuron for
  hidden and optionally output bias currents.
- `NMNIST_ACTIVITY_LAMBDA=<float>`: adds a hidden-activation penalty during
  sparse finetune to trade accuracy for fewer hidden spikes/deliveries.
- `NMNIST_THR_MIN`, `NMNIST_THR_MAX`, `NMNIST_THR_STEP`: threshold search range.
- `NMNIST_THRESHOLD_OBJECTIVE=accuracy|ops|edge`: threshold selection objective.
  `edge` uses `NMNIST_THRESHOLD_OP_PENALTY` to mildly penalize delivery count.

Current benchmark targets:

- `mnist`: frame MNIST rate-coded SNN sanity check.
- `sparse`: controlled sparse-event microbenchmark.
- `shd`: public spike-audio dataset; hard for the current count model.
- `nmnist`: public event-camera digit dataset.

Suggested benchmark set for architecture reporting:

```bash
# Controlled best-case sparse-event scaling sanity check.
python3 tools/bench.py export sparse
python3 tools/bench.py run sparse --design event_ram --tag _sparse_report --max-cycles 100000000
python3 tools/bench.py run sparse --design ann --tag _sparse_report
bash tools/run_yosys_synth.sh _sparse_report event_ram xilinx
python3 tools/report.py --tag _sparse_report --event-design event_ram

# N-MNIST efficiency point: strong storage/op/area win, lower accuracy.
python3 tools/bench.py export nmnist \
  --set NMNIST_T_STEPS=50 \
  --set NMNIST_TOPK_W1=128 \
  --set NMNIST_FINETUNE_EPOCHS=12
python3 tools/bench.py run nmnist --design event_ram --tag _nmnist_t50_k128_bram --max-cycles 300000000
python3 tools/bench.py run nmnist --design ann --tag _nmnist_t50_k128_bram
bash tools/run_yosys_synth.sh _nmnist_t50_k128_bram event_ram xilinx
bash tools/run_yosys_synth.sh _nmnist_t50_k128_bram ann xilinx
bash tools/run_yosys_synth.sh _nmnist_t50_k128_bram event_ram asic
bash tools/run_yosys_synth.sh _nmnist_t50_k128_bram ann asic
python3 tools/report.py --tag _nmnist_t50_k128_bram --event-design event_ram

# N-MNIST higher-accuracy sparse point.
python3 tools/bench.py export nmnist \
  --set NMNIST_T_STEPS=50 \
  --set NMNIST_TOPK_W1=192 \
  --set NMNIST_FINETUNE_EPOCHS=20
python3 tools/bench.py run nmnist --design event_ram --tag _nmnist_t50_k192_e20_bram --max-cycles 400000000
python3 tools/bench.py run nmnist --design ann --tag _nmnist_t50_k192_e20_bram
bash tools/run_yosys_synth.sh _nmnist_t50_k192_e20_bram event_ram xilinx
bash tools/run_yosys_synth.sh _nmnist_t50_k192_e20_bram ann xilinx
bash tools/run_yosys_synth.sh _nmnist_t50_k192_e20_bram event_ram asic
bash tools/run_yosys_synth.sh _nmnist_t50_k192_e20_bram ann asic
python3 tools/report.py --tag _nmnist_t50_k192_e20_bram --event-design event_ram
```
