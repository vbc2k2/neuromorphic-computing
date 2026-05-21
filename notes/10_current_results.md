# Current Credible Results

## Balanced N-MNIST: `_nmnist_t50_k128`

Configuration:

- Dataset: N-MNIST
- Selection: balanced, 300 exported test samples, 30 per class
- Network: 2312 input channels, 128 hidden neurons, 10 outputs
- Timesteps: 50
- Sparse W1: top-128 input weights per hidden neuron
- Sparse fine-tuning: 12 epochs

Measured RTL results:

| Design | Accuracy | Active cycles | Ops | Model storage |
| --- | ---: | ---: | ---: | ---: |
| INT8 ANN | 86.67% | 89,247,600 | 89,164,800 MACs | 297,216 dense weights |
| Event SNN | 81.67% | 16,012,711 | 12,516,577 synapse deliveries | 17,391 CSR synapses |

Ratios:

- SNN accuracy gap: -5.00 percentage points
- SNN active-cycle reduction: 5.57x
- SNN operation reduction: 7.12x
- SNN model-storage reduction: 17.09x

Yosys `synth_xilinx -noiopad` area estimates:

| Design | Estimated LCs | BRAM | DSP |
| --- | ---: | ---: | ---: |
| INT8 ANN | 30,507 | not inferred in current report | 3 DSP48E1 |
| Spatial event SNN | 209,784 | 8 RAMB36E1 | 0 DSP48E1 |

Interpretation:

- The current spatial event SNN has strong work/storage reductions but poor FPGA area.
- The area loss is architectural: 2451 physical `neuron_core` instances plus wide per-neuron routing, priority, and popcount logic.
- The v3 direction is a RAM-state event processor with one or a few shared neuron update lanes.
