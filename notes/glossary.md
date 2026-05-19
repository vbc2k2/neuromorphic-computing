# Glossary

## Active Cycles

Cycles where the accelerator is doing useful work instead of sitting idle.

## ANN

Artificial neural network. In this project, the ANN is the floating-point
PyTorch model trained before SNN conversion.

## Argmax

The index of the largest value in a list. For classification, the predicted
digit is the output index with the largest value or spike count.

## Bias

A trainable offset added to a neuron's weighted sum. In the SNN RTL, bias is
implemented as a special neuron that fires every timestep.

## CSR

Compressed sparse row. A compact way to store sparse connections. This project
groups synapses by source neuron.

## Dense Baseline

The comparison design that scans every synapse every timestep. It is dense in
scheduling, even though it uses the same CSR files.

## DUT

Design under test. The RTL module being simulated by a testbench.

## Event-Driven

A computation style where work happens in response to events. In this project,
the events are spikes.

## Fan-Out

The set of outgoing connections from one source neuron to its destination
neurons.

## FSM

Finite-state machine. Hardware control logic that moves through named states.

## Golden Model

A reference implementation used to define correct behavior. Here, the Python
integer SNN simulation is the golden model for RTL.

## LIF

Leaky integrate-and-fire. A neuron model with membrane potential, leak,
integration, threshold, fire, and reset.

## Membrane Potential

The internal state of a spiking neuron. Incoming weighted spikes increase or
decrease it. Leak gradually reduces it.

## MNIST

A handwritten digit dataset containing 28 by 28 grayscale images.

## Quantization

Mapping floating-point values into lower-precision integer values. This project
quantizes weights and biases to signed 8-bit integers.

## Rate Coding

Representing a value by spike frequency over time. Brighter pixels produce more
spikes.

## RTL

Register-transfer level. A hardware description style that models registers,
combinational logic, memories, and state machines.

## SLURM

A cluster job scheduler used to submit and manage compute jobs.

## SNN

Spiking neural network. A neural network that communicates using spike events
over timesteps.

## Spike

A binary firing event from a neuron.

## Synapse

A weighted connection from one neuron to another.

## Testbench

Simulation code that drives the DUT, provides inputs, checks outputs, and
writes logs or CSV files.

## Threshold

The membrane potential level required for a neuron to fire.

## Timestep

One discrete simulation step of the SNN.

## Xcelium

Cadence's HDL simulator. The server scripts run it using `xrun`.

