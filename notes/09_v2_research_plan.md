# Chapter 9: V2 Research Plan

## 9.1 Why V2 Exists

The v1 project compares:

```text
event-driven SNN vs dense SNN
```

That is a valid architecture comparison, but it does not answer a bigger
question:

```text
Is this better than a conventional quantized ANN accelerator?
```

V2 begins answering that by adding a third baseline:

```text
INT8 ANN baseline
```

The v2 comparison becomes:

```text
event-driven SNN
dense SNN
INT8 ANN
```

## 9.2 What The INT8 ANN Baseline Is

The ANN baseline runs a normal dense MLP:

```text
raw uint8 pixels -> int8 W1 -> ReLU uint8 hidden -> int8 W2 -> argmax
```

It uses the same trained model family as the SNN conversion. It is not
spike-based and does not run for many timesteps.

This baseline matters because most commercial NPUs are closer to ANN tensor
accelerators than SNN accelerators.

## 9.3 New Files

Python export:

```text
ann_pixels.mem
ann_w1.mem
ann_b1.mem
ann_w2.mem
ann_b2.mem
```

RTL:

```text
rtl/top_ann.sv
tb/tb_classify_ann.sv
```

Server:

```text
server/sbatch_mnist_v2.sh
```

Results:

```text
python/merge_results_v2.py
results/mnist_v2_comparison.csv
results/mnist_v2_comparison.png
```

## 9.4 How To Run V2

Regenerate the Python exports first:

```bash
python3 python/train_snn.py
```

Then submit the v2 server job:

```bash
sbatch server/sbatch_mnist_v2.sh
```

After all slices finish:

```bash
python3 python/merge_results_v2.py
```

## 9.5 Why This Is More Research-Level

V1 can claim:

```text
The event-driven SNN does less synaptic work than a dense SNN baseline.
```

V2 can start asking:

```text
How does event-driven SNN inference compare against conventional INT8 ANN
inference?
```

That is a stronger and more honest question.

## 9.6 What V2 Still Does Not Prove

V2 does not yet prove ASIC energy efficiency.

For that, the project still needs:

```text
synthesis
timing
area
power
energy per inference
memory traffic estimates
```

Vivado FPGA synthesis can provide useful first evidence:

```text
LUTs
FFs
BRAMs
DSPs
timing
estimated FPGA power
```

ASIC flow would be needed for stronger silicon-level claims.

## 9.7 Research Roadmap

Recommended next steps:

1. Finish v1 server results and freeze the baseline.
2. Run v2 on the same MNIST export.
3. Compare event SNN, dense SNN, and INT8 ANN.
4. Add Vivado synthesis reports.
5. Add N-MNIST or another native event workload.
6. Add low-power architectural optimizations:

```text
clock gating
smaller bit widths
event FIFO sizing
CSR compression
early exit
memory banking
weight pruning
```

7. Turn the accelerator into a memory-mapped peripheral.
8. Attach it to a RISC-V SoC such as CVA6.

## 9.8 HPRC/Open-Source Simulation Strategy

For open-source reproducibility, prefer this split:

```text
HPRC:
  Python training/export sweeps
  Verilator lint
  Verilator smoke runs for open-source simulation

TAMU Xcelium server:
  reference RTL runs
  compatibility checks against commercial simulation

Vivado:
  FPGA synthesis, utilization, timing, and estimated power
```

HPRC scratch is not backed up, so keep source on GitHub and regenerate large
artifacts:

```text
data/mnist.npz
sim/*.mem
sim/*.csv
obj_dir/
```

The MNIST loader in `python/train_snn.py` is TensorFlow-free. It downloads the
same Keras MNIST `.npz` file directly with `urllib` and loads it with NumPy.
This avoids TensorFlow/PyTorch native-library conflicts on HPC nodes.

The script also defaults common CPU math thread variables to 1:

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
```

Override them only inside an allocated job when the scheduler grants more
cores.

After generating the memories on HPRC, run Verilator checks:

```bash
bash tools/verilator_lint.sh
```

Run a fast INT8 ANN smoke test:

```bash
bash tools/run_verilator_ann.sh 0 4 _smoke
```

Run a fast event-driven SNN smoke test:

```bash
bash tools/run_verilator_event.sh 0 4 _smoke
```

That produces:

```text
sim/classify_ann_smoke.csv
sim/metrics_classify_ann_smoke.csv
sim/classify_event_smoke.csv
sim/metrics_classify_event_smoke.csv
```

For longer HPRC runs, keep `VERILATOR_JOBS` within the core count granted by
SLURM. The event SNN is expected to run much slower than the ANN on frame MNIST;
that result is part of the point of v2 because it shows why sparse/event-native
benchmarks are needed.

## 9.9 Sparse Event Microbenchmark

MNIST is a dense frame benchmark. Rate coding it into 50 timesteps makes the
SNN do too much work. To test the opposite case, the repo includes a controlled
sparse-event benchmark:

```bash
python3 python/export_sparse_event.py
bash tools/run_verilator_event.sh 0 299 _sparse
bash tools/run_verilator_ann.sh 0 299 _sparse
python3 python/merge_results_v2.py
```

This overwrites the files in `sim/` with sparse-event memories. To return to
MNIST, rerun:

```bash
python3 python/train_snn.py
```

The sparse benchmark is not a public dataset result. It is a microbenchmark for
the architectural claim: event-driven SNN execution can avoid work when the
input stream and connectivity are genuinely sparse. A publishable version still
needs a real event dataset such as N-MNIST, DVS Gesture, keyword spotting, or
sensor anomaly detection.

## 9.10 SHD Public Spike-Audio Benchmark

SHD is the first public dataset path to try because it is already spike-based
and has only 700 input channels, which fits the current fully connected RTL
better than raw 128x128 DVS vision.

Install the dataset loader:

```bash
python3 -m pip install tonic h5py
```

Export a trained sparse SHD model and run both RTL baselines:

```bash
python3 python/export_shd.py
bash tools/run_verilator_event.sh 0 299 _shd
bash tools/run_verilator_ann.sh 0 299 _shd
cat sim/metrics_classify_event_shd.csv
cat sim/metrics_classify_ann_shd.csv
```

This is a real public dataset, but the first model is still intentionally small:
a count-based MLP is trained, pruned, quantized, and converted to the same event
RTL format. Treat the result as the first hardware-sparsity probe on public
data. If accuracy and work reduction look good, the next step is a stronger
temporal model rather than only count features.

## 9.11 N-MNIST Public Event-Vision Benchmark

If SHD is too hard for the current count-based model, N-MNIST is the next public
benchmark to try. It is event-camera digit data, so it is easier than SHD while
still being a public event-stream dataset.

```bash
python3 -m pip install tonic
python3 python/export_nmnist.py
bash tools/run_verilator_event.sh 0 299 _nmnist
bash tools/run_verilator_ann.sh 0 299 _nmnist
cat sim/metrics_classify_event_nmnist.csv
cat sim/metrics_classify_ann_nmnist.csv
```

As with SHD, inspect the export accuracies first. Do not spend RTL time if the
export reports poor sparse INT8 ANN or sparse SNN accuracy.

## 9.12 Strong Final Claim To Aim For

Do not claim:

```text
SNNs beat all NPUs.
```

Aim for:

```text
For sparse edge workloads, an open-source event-driven SNN accelerator can
reduce work and estimated energy versus dense SNN execution while remaining
competitive with an INT8 ANN baseline under defined accuracy and latency
constraints.
```
