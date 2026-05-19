# Neuromorphic Computing Project Notes

These notes are a textbook-style guide for this repository. They start from
basic ideas and build up to the exact Python, RTL, simulation, and server flow
used by the project.

## Recommended Reading Order

1. `00_learning_path.md`
   - What this project is trying to prove.
   - How the chapters fit together.

2. `01_digital_logic_and_rtl.md`
   - Digital logic basics.
   - Registers, clocks, finite-state machines, memories, and RTL thinking.

3. `02_neural_network_basics.md`
   - What a neural network is.
   - Layers, weights, bias, activation, training, and accuracy.

4. `03_spiking_neural_networks.md`
   - Why SNNs use spikes instead of continuous values.
   - Rate coding, LIF neurons, timesteps, thresholding, and spike counts.

5. `04_project_pipeline.md`
   - The full project flow from Python training to RTL simulation.
   - How the files in `python/`, `rtl/`, `tb/`, `sim/`, and `server/` connect.

6. `05_python_training_export.md`
   - What `python/train_snn.py` does.
   - Quantization, CSR export, spike export, and Verilog config generation.

7. `06_rtl_architecture.md`
   - The SystemVerilog architecture.
   - `neuron_core`, `synapse_csr`, `spike_router`, `top`, and `top_dense`.

8. `07_server_slurm_xcelium.md`
   - How the university server run works.
   - SLURM array jobs, Xcelium, slices, logs, and result merging.

9. `08_debugging_results.md`
   - How to read logs.
   - Common failures and what they mean.
   - How to validate that results are real.

10. `glossary.md`
    - Short definitions of project terms.

11. `09_v2_research_plan.md`
    - Research-level roadmap.
    - Adds the INT8 ANN baseline and future low-power accelerator work.

## What You Should Be Able To Explain After Reading

- Why this project compares an event-driven accelerator against a dense
  clock-driven baseline.
- How a normal ANN is trained and then converted into an integer SNN.
- Why the RTL and Python model must use the same integer neuron math.
- What CSR means and why it is useful for sparse event-driven computation.
- Why the event-driven design does less work while producing the same
  predictions as the dense baseline.
- How to run the server job and know whether the logs are valid.
- Why v2 adds a conventional INT8 ANN baseline before making stronger claims.

## Project In One Paragraph

The project trains a handwritten-digit classifier in Python, converts it into a
spiking neural network, exports the network weights and input spikes into memory
files, and runs two hardware designs on the same workload. The event-driven
design only processes synapses for neurons that actually spike. The dense
baseline scans every synapse every timestep. If both designs produce the same
classification outputs, then the project can compare their work metrics fairly.
