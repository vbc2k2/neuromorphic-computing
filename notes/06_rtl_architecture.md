# Chapter 6: RTL Architecture

## 6.1 Main RTL Modules

The core RTL modules are:

```text
neuron_core.sv
synapse_csr.sv
spike_router.sv
top.sv
top_dense.sv
```

The testbenches are:

```text
tb_classify.sv
tb_classify_dense.sv
tb_neuron_core.sv
```

## 6.2 `neuron_core.sv`

`neuron_core` implements the integer LIF neuron.

Inputs:

- Incoming spike valid.
- Incoming spike weight.
- Timestep tick.
- Clock and reset.

State:

- Membrane potential.
- Accumulated weighted input.

Output:

- Spike pulse when threshold is reached.

Conceptually:

```text
accumulate incoming weights during the timestep
on timestep tick:
    leak membrane
    add accumulated input
    compare against threshold
    fire or keep membrane
```

This module is shared by both event-driven and dense designs.

## 6.3 `synapse_csr.sv`

`synapse_csr` stores the network connectivity.

It has two interfaces:

1. Pointer lookup:

```text
source neuron ID -> base index and fan-out length
```

2. Synapse read:

```text
flat synapse index -> destination neuron ID and weight
```

It loads:

```text
csr_dst.mem
csr_weight.mem
csr_start.mem
csr_count.mem
```

This module is also shared by both designs.

## 6.4 Why CSR Is Useful

Without CSR, a source neuron might need a full row of possible connections to
every target.

With CSR, each source has a compact list of actual outgoing synapses.

For a spiking source:

```text
read start and count
walk only that range
deliver each listed synapse
```

This is exactly what event-driven hardware needs.

## 6.5 `spike_router.sv`

`spike_router` exists only in the event-driven design.

It accepts source neuron IDs that spiked. It stores them in a FIFO. For each
source ID, it:

1. Looks up that source in CSR.
2. Reads each outgoing synapse.
3. Delivers weight to the target neuron.

Important counters:

- `total_events`: number of source spike events processed.
- `total_deliveries`: number of synapse deliveries performed.

`total_deliveries` is the event-driven version of synapse operations.

## 6.6 `top.sv`

`top.sv` is the event-driven accelerator.

It connects:

```text
external spike input
-> spike router
-> CSR memory
-> neuron array
-> fired-neuron feedback
```

For each timestep:

1. Accept external input spikes while idle.
2. Route all queued spike events.
3. Tick all neurons.
4. Capture fired neurons.
5. Queue fired neurons as events for the next timestep.

The event-driven design's work depends on how many spikes occur.

## 6.7 `top_dense.sv`

`top_dense.sv` is the dense baseline.

It does not use the spike router. Instead, it scans the full CSR every
timestep.

For each timestep:

```text
for each source neuron:
    for each outgoing synapse:
        count one synapse operation
        deliver weight only if the source spiked
```

The dense baseline counts every synapse scan as work, even if the source neuron
did not spike.

That is why dense work scales with network size, not spike activity.

## 6.8 Why Dense Still Uses CSR

The dense baseline could have used a matrix, but using CSR keeps the weights and
connectivity identical to the event-driven design.

The dense design is "dense" in scheduling, not in file format.

It scans all stored synapses every timestep.

## 6.9 Testbench Classification Flow

The classification testbenches do:

```text
read snn_spikes.mem
read snn_labels.mem
open result CSV
for each image:
    reset DUT
    for each timestep:
        inject input spikes
        inject bias spike
        run one timestep
    prediction = output neuron with max spike count
    write CSV row
write metrics CSV
finish
```

The testbench handles image sequencing and CSV output. The DUT handles the
accelerator behavior.

## 6.10 Performance Counters

Event-driven metrics:

```text
active_cycles
router_events
router_deliveries
total_spikes_fired
```

Dense metrics:

```text
active_cycles
total_synapse_ops
total_updates
total_spikes_fired
```

The comparison usually focuses on:

```text
event router_deliveries vs dense total_synapse_ops
```

## 6.11 Functional Equivalence

Functional equivalence means both designs produce the same predictions.

They should match because they share:

- Same generated input spikes.
- Same labels.
- Same CSR connectivity.
- Same weights.
- Same `neuron_core`.
- Same output spike counting rule.

If predictions differ, debug correctness before discussing performance.

## 6.12 Exercise

1. What does `neuron_core` store internally?
2. What does `synapse_csr` return for a source neuron?
3. Why does `spike_router` need a FIFO?
4. What is the main scheduling difference between `top` and `top_dense`?
5. Which metric best represents event-driven synapse work?

