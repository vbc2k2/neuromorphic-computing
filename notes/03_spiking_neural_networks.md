# Chapter 3: Spiking Neural Networks

## 3.1 From Numbers To Events

A normal ANN passes numerical values between layers.

An SNN passes spikes.

A spike is a binary event:

```text
spike = 1 means the neuron fired
spike = 0 means it did not fire
```

Instead of saying "this pixel has value 180", an SNN might say "this input
neuron fired in many timesteps."

## 3.2 Time Matters

SNNs simulate over timesteps.

This project uses:

```text
T_STEPS = 50 for MNIST
```

For each image, the network runs for 50 timesteps. Output spikes are counted
over those timesteps.

## 3.3 Rate Coding

Rate coding represents a value by how often a spike occurs.

Bright pixel:

```text
many spikes over 50 timesteps
```

Dark pixel:

```text
few or no spikes over 50 timesteps
```

So a pixel value becomes a spike train.

Example:

```text
pixel value high:
    1 1 0 1 1 0 1 1 ...

pixel value low:
    0 0 0 1 0 0 0 0 ...
```

## 3.4 Input Spike Memory

The generated file `sim/snn_spikes.mem` stores input spikes.

For each image and timestep, it stores one binary word containing all input
pixel spikes.

For MNIST:

```text
one word = 784 bits
```

The testbench reads this word and injects spikes for the pixels whose bits are
1.

## 3.5 LIF Neuron

The neuron model is LIF:

```text
leaky integrate-and-fire
```

It has a membrane potential. You can think of membrane potential as stored
charge.

Each timestep:

1. Leak reduces the membrane potential.
2. Incoming weighted spikes are integrated.
3. If the membrane reaches threshold, the neuron fires.
4. If it fires, the membrane resets.

## 3.6 Integer LIF Math

The simplified math in this project is:

```text
leaked = membrane - LEAK, but do not cross positive values below zero
integrated = leaked + weight_sum
fired = integrated >= threshold
if fired:
    membrane = RESET_VAL
else:
    membrane = integrated
```

The key requirement is that Python and RTL must implement the same integer math.

If the Python golden model and `neuron_core.sv` differ, the hardware comparison
is not meaningful.

## 3.7 Weighted Spikes

A spike itself is just an event, but each synapse has a weight.

If source neuron `s` spikes, and it connects to target neuron `t` with weight
`w`, then target neuron `t` receives `w`.

Multiple incoming spikes add together.

```text
weight_sum = w0 + w1 + w2 + ...
```

## 3.8 Bias Spike

The project uses a bias neuron.

The bias neuron fires every timestep. It connects to hidden and output neurons
with the quantized bias values from the trained ANN.

This converts ANN bias into SNN synapses.

## 3.9 Output Decision

The SNN output layer has 10 output neurons.

During the 50 timesteps, the testbench counts output spikes:

```text
count[0], count[1], ..., count[9]
```

The predicted digit is:

```text
prediction = argmax(count)
```

If output neuron 3 fires the most times, the SNN predicts digit 3.

## 3.10 Threshold

Threshold controls how much membrane potential is needed before a neuron fires.

If threshold is too low:

- Too many neurons fire.
- The network may become noisy.

If threshold is too high:

- Too few neurons fire.
- The network may become silent.

`train_snn.py` searches over threshold values and picks the one with best
accuracy on a tuning subset.

## 3.11 Sparsity

Sparsity means many values are zero or inactive.

In an SNN, sparsity means many neurons do not spike at a given timestep.

This is the foundation of the event-driven accelerator:

```text
if a neuron did not spike, do not walk its outgoing synapses
```

## 3.12 Event-Driven vs Dense

Dense baseline:

```text
for every timestep:
    scan every synapse
```

Event-driven:

```text
for every timestep:
    for each spiking source neuron:
        scan only that source neuron's outgoing synapses
```

If few neurons spike, event-driven work is much smaller.

## 3.13 Exercise

1. What is a spike?
2. Why does an SNN need timesteps?
3. How does rate coding represent a bright pixel?
4. What does threshold do?
5. Why can sparse spikes reduce hardware work?

