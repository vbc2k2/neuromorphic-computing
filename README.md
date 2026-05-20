# Event-Driven SNN Accelerator — Handwritten Digit Classifier

A **spiking neural network (SNN) accelerator** in SystemVerilog that classifies
handwritten digits, with a dense clock-driven baseline for a rigorous
architectural comparison.

The headline result: the **event-driven** accelerator and the **dense baseline**
are *functionally equivalent* — bit-identical predictions, identical accuracy —
but the event-driven design reaches that result with far fewer synaptic
operations and active cycles, because it only does work for neurons that
actually spike.

| | Phase 1 — 8×8 digits | Phase 2 — MNIST |
|---|---|---|
| Network | 64→64→10 (139 neurons, 4.8k synapses) | 784→256→10 (1051 neurons, 203k synapses) |
| Where it runs | locally, Vivado xsim | compute server, Cadence Xcelium + SLURM |
| Status | ✅ done & verified | flow ready — run on the server |

## Phase 1 result — 8×8 digits, 60 test images

| Metric | Event-Driven | Dense Baseline | |
|---|---|---|---|
| Classification accuracy | 93.33% | 93.33% | identical → equivalent |
| Synaptic operations | 3,886,662 | 11,544,000 | event **2.97× fewer** |
| Active cycles (work) | 15,821,217 | 47,184,000 | event **2.98× fewer** |

Both designs produce bit-identical predictions on every image. The numpy integer
SNN model (the golden spec) scores **96.4%** on the full digits test set; the RTL
reproduces its 8-bit-quantized accuracy exactly.

## Phase 2 — MNIST (server)

`train_snn.py` (`DATASET = 'mnist'`) trains a 784→256→10 MLP, converts it to a
rate-coded SNN, and exports the same CSR connectivity + spike-train format the
RTL already uses — only larger. Verified offline:

- ANN test accuracy: **98.22%**
- Quantized int8 SNN: **97.05%** (full 10k MNIST test set)

The RTL is fully parameterized, so the same five modules run the MNIST network
unchanged. Because the dense baseline does ~203k synaptic ops *every timestep*,
MNIST is too heavy for xsim — it runs on a server (see [`server/`](server/)),
with the test set split across cores by a SLURM array job.

## How it works

1. **Train** (`python/train_snn.py`): an MLP is trained on the dataset,
   converted to a rate-coded SNN, quantized to signed 8-bit, and verified by an
   integer numpy simulation that mirrors the RTL neuron exactly (the golden spec).
2. **Export**: connectivity is written as a **CSR** (compressed-sparse-row)
   store grouped by source neuron; test images become input spike trains.
3. **Classify** (RTL): for each image, rate-coded spikes are fed in for T
   timesteps; the output neuron that spikes most is the predicted digit.

Both accelerators run the SAME network on the SAME `neuron_core`, so any
difference in behaviour is impossible by construction — only the *amount of
work* differs:

| | Event-driven (`top.sv`) | Dense baseline (`top_dense.sv`) |
|---|---|---|
| Per timestep | walks the CSR fan-out of **only the neurons that spiked** | walks **every synapse** of the network |
| Work scales with | spike activity | network size |

## Architecture

| Module | Description |
|---|---|
| `neuron_core.sv` | Shared discrete-timestep LIF neuron (leak → integrate → threshold) |
| `synapse_csr.sv` | CSR connectivity store — synapses grouped by source neuron |
| `spike_router.sv` | Event-driven router — walks a spiking neuron's fan-out in O(fan-out) |
| `top.sv` | Event-driven accelerator top-level |
| `top_dense.sv` | Dense clock-driven baseline top-level |

## Quick Start

### Unified benchmark CLI

For the current v2 flow, use the benchmark helper instead of manually
remembering each export/run command:

```bash
source tools/env_hprc.sh          # HPRC: conda + Verilator PATH helper
python3 tools/bench.py doctor    # check Python deps, Verilator, current sim config
python3 tools/bench.py list      # show available benchmarks

python3 tools/bench.py all sparse
python3 tools/bench.py all nmnist
```

Step-by-step:

```bash
python3 tools/bench.py export nmnist
python3 tools/bench.py run nmnist --design event --last 49
python3 tools/bench.py run nmnist --design ann --last 49
python3 tools/bench.py summarize nmnist
```

See [`tools/README.md`](tools/README.md) for tuning options and benchmark
targets.

### Phase 1 — digits, locally (Vivado xsim)
```batch
REM python/train_snn.py must have DATASET = 'digits'
cd sim
run_all.bat        REM train -> unit test -> event -> dense -> compare
```
Individual stages: `run_neuron_test.bat`, `run_classify.bat`,
`run_classify_dense.bat`, then `python ..\python\compare_classify.py`.

### Phase 2 — MNIST, on the server (Xcelium + SLURM)
```bash
# python/train_snn.py with DATASET = 'mnist'
python python/train_snn.py          # export MNIST CSR + spikes + config
sbatch server/sbatch_mnist.sh       # run both designs, sliced across cores
python python/merge_results.py      # merge -> results/mnist_comparison.png
```
See [`server/README.md`](server/README.md) for details.

## Project Structure

```
neuromorphic_computing/
├── rtl/
│   ├── neuron_core.sv      # shared LIF neuron
│   ├── synapse_csr.sv      # CSR connectivity store
│   ├── spike_router.sv     # event-driven sparse router
│   ├── top.sv              # event-driven accelerator
│   └── top_dense.sv        # dense baseline accelerator
├── tb/
│   ├── tb_neuron_core.sv   # neuron unit test
│   ├── tb_classify.sv      # event-driven classification testbench
│   └── tb_classify_dense.sv# dense baseline classification testbench
├── python/
│   ├── train_snn.py        # train + ANN->SNN convert + quantize + export
│   ├── compare_classify.py # Phase 1 accuracy + efficiency comparison
│   └── merge_results.py    # merge sliced server results (Phase 2)
├── server/                 # Xcelium + SLURM scripts for the MNIST run
├── sim/                    # run scripts; generated weights/spikes/config
├── results/                # generated comparison plots + CSVs
└── README.md
```

## Author

Neuromorphic Accelerator Project — RTL implementation and architectural
comparison of an event-driven vs dense spiking neural network accelerator,
benchmarked on handwritten digit classification (8×8 digits and MNIST).
