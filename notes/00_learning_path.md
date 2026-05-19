# Chapter 0: Learning Path

## 0.1 What This Project Is

This repository implements a spiking neural network accelerator in
SystemVerilog. The accelerator classifies handwritten digits. The important
part is not only classification accuracy. The important part is architectural
comparison.

The project compares two hardware designs:

- Event-driven design: does work only when spikes occur.
- Dense baseline: scans every synapse every timestep.

Both designs use the same network, same weights, same input spike trains, and
same neuron core. That makes the comparison meaningful. If the predictions
match, then differences in work metrics come from architecture, not from model
behavior.

## 0.2 The Big Idea

A normal neural network computes with numbers. A spiking neural network computes
with events called spikes. If very few neurons spike at a given time, then an
event-driven accelerator can skip a lot of work.

This project asks:

```text
Can we get the same classification result while doing fewer synaptic
operations?
```

The answer should be yes when spike activity is sparse.

## 0.3 What You Need To Know

The project touches four areas:

1. Neural networks
   - Layers, weights, biases, activations, training, and accuracy.

2. Spiking neural networks
   - Rate-coded input, timesteps, membrane potential, threshold, reset, and
     output spike counts.

3. Digital hardware
   - Clocks, registers, memories, finite-state machines, modules, and
     simulation.

4. Server execution
   - Python dependencies, Xcelium, SLURM array jobs, log files, and result
     merging.

You do not need to master all four before understanding the project. The most
useful path is:

```text
ANN basics -> SNN basics -> exported memory files -> RTL modules -> server flow
```

## 0.4 Repository Map

```text
python/
  train_snn.py          Train ANN, convert to SNN, export memory files.
  compare_classify.py   Compare local event/dense results.
  merge_results.py      Merge server slice results.

rtl/
  neuron_core.sv        Shared LIF neuron.
  synapse_csr.sv        CSR synapse memory.
  spike_router.sv       Event-driven synapse walker.
  top.sv                Event-driven accelerator.
  top_dense.sv          Dense baseline accelerator.

tb/
  tb_classify.sv        Testbench for event-driven classifier.
  tb_classify_dense.sv  Testbench for dense baseline classifier.
  tb_neuron_core.sv     Unit test for the neuron.

sim/
  csr_*.mem             Generated network connectivity.
  snn_spikes.mem        Generated input spike trains.
  snn_labels.mem        Generated labels.
  snn_config.vh         Generated Verilog constants.
  classify_*.csv        Simulation classification outputs.
  metrics_*.csv         Simulation work metrics.

server/
  sbatch_mnist.sh       SLURM array job.
  xcelium_run.sh        Compile and run one design slice using Xcelium.
```

## 0.5 How To Study The Project

Read the notes in this order:

1. Understand what a clocked hardware design is.
2. Understand what a trained neural network is.
3. Understand how a spiking neuron turns weighted inputs into spikes.
4. Understand why CSR stores only real connections.
5. Understand why event-driven routing avoids dense scanning.
6. Understand how the server splits 300 MNIST images into six slices.
7. Learn to read logs and reject invalid simulation results.

## 0.6 Mental Model

Think of each image classification as a short simulation:

```text
for each image:
    reset all neuron states
    for each timestep:
        inject input spikes from the image
        inject the bias spike
        route spikes through synapses
        update neurons
        count output spikes
    prediction = output neuron with highest spike count
```

The event-driven design and dense design both follow this idea. They differ in
how they route synaptic work.

## 0.7 What Counts As Success

A successful full run should show:

- No `$readmem` errors.
- Event and dense CSVs exist for every slice.
- Event and dense predictions match.
- Accuracy is close to the Python golden SNN result for the exported images.
- Dense synapse operations are much larger than event-driven synapse
  deliveries.

If any of those fail, the result is not yet a valid architecture comparison.

