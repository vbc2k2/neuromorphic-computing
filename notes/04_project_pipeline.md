# Chapter 4: Project Pipeline

## 4.1 Full Flow

The full project flow is:

```text
1. Train ANN in Python.
2. Quantize ANN weights and biases.
3. Convert input images into spike trains.
4. Simulate integer SNN in Python as golden model.
5. Export memory files for RTL.
6. Run event-driven RTL.
7. Run dense baseline RTL.
8. Compare predictions and work metrics.
```

Each stage depends on the previous one.

## 4.2 Python Produces The Hardware Inputs

`python/train_snn.py` writes files into `sim/`:

```text
csr_dst.mem
csr_weight.mem
csr_start.mem
csr_count.mem
snn_spikes.mem
snn_labels.mem
snn_config.vh
```

The RTL testbenches load these files.

If these files are missing or stale, the hardware simulation is not running the
network you think it is running.

## 4.3 The Generated Verilog Config

`sim/snn_config.vh` contains constants:

```text
SNN_N_TOTAL
SNN_N_INPUT
SNN_N_HIDDEN
SNN_N_OUTPUT
SNN_T_STEPS
SNN_THRESHOLD
SNN_NUM_TEST
SNN_NUM_SYN
```

The testbenches include this file:

```systemverilog
`include "snn_config.vh"
```

That lets the same RTL scale from the small digits network to the larger MNIST
network.

## 4.4 Network Layout

The project assigns every neuron an integer ID.

For MNIST:

```text
0..783      input pixel neurons
784         bias neuron
785..1040   hidden neurons
1041..1050  output neurons
```

This layout is important because CSR files store source and destination neuron
IDs.

## 4.5 What CSR Stores

CSR means compressed sparse row. In this project, it stores synapses grouped by
source neuron.

Files:

```text
csr_dst.mem     destination neuron ID for each synapse
csr_weight.mem  signed weight for each synapse
csr_start.mem   starting synapse index for each source neuron
csr_count.mem   number of outgoing synapses for each source neuron
```

If source neuron `s` spikes:

```text
start = csr_start[s]
count = csr_count[s]
for j from start to start + count - 1:
    dst = csr_dst[j]
    weight = csr_weight[j]
    deliver weight to dst
```

That is the core operation of the event-driven design.

## 4.6 Event-Driven Run

The event-driven testbench is `tb/tb_classify.sv`.

It instantiates `rtl/top.sv`, which contains:

- CSR memory.
- Spike router.
- Neuron array.
- Timestep FSM.

The output CSV is:

```text
sim/classify_event_<slice>.csv
sim/metrics_classify_event_<slice>.csv
```

## 4.7 Dense Baseline Run

The dense testbench is `tb/tb_classify_dense.sv`.

It instantiates `rtl/top_dense.sv`, which:

- Uses the same CSR memory.
- Uses the same neuron core.
- Scans all synapses every timestep.

The output CSV is:

```text
sim/classify_dense_<slice>.csv
sim/metrics_classify_dense_<slice>.csv
```

## 4.8 Why Both Designs Use The Same CSR

Using the same CSR store avoids an unfair comparison.

If event-driven and dense designs used different memory formats or different
weights, mismatched results could come from data differences.

Here they share the same connectivity and neuron math, so the comparison is
cleaner.

## 4.9 What The Server Scripts Do

For MNIST, running all 300 exported images in one simulation can be slow. The
server flow splits the work into six slices:

```text
slice 0: images 0..49
slice 1: images 50..99
slice 2: images 100..149
slice 3: images 150..199
slice 4: images 200..249
slice 5: images 250..299
```

Each slice runs:

```text
event design
dense design
```

After all slices finish, `python/merge_results.py` merges them.

## 4.10 Valid Result Checklist

Before trusting results, check:

- `train_snn.py` finished and exported files.
- `snn_config.vh` matches the intended dataset.
- All six SLURM slices completed.
- No `$readmem` errors in logs.
- Event and dense CSVs exist for every slice.
- Event and dense predictions match.
- Metrics are nonzero.

## 4.11 Exercise

1. Which Python file exports the RTL memory files?
2. Which file tells RTL how many neurons and synapses exist?
3. Why are images split into slices on the server?
4. Why is it useful that both RTL designs share `neuron_core.sv`?
5. What files should exist before running `merge_results.py`?

