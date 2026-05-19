# Chapter 5: Python Training And Export

## 5.1 Role Of `train_snn.py`

`python/train_snn.py` is the bridge between machine learning and RTL.

It does five jobs:

1. Load dataset.
2. Train a floating-point ANN.
3. Quantize the ANN into integer weights.
4. Simulate the integer SNN golden model.
5. Export files used by SystemVerilog testbenches.

## 5.2 Dataset Selection

Near the top of the file:

```python
DATASET = 'mnist'
```

The project supports:

```text
digits -> small 8x8 sklearn digits dataset
mnist  -> full 28x28 MNIST dataset
```

For local fast simulation, `digits` is easier.

For the server benchmark, use `mnist`.

## 5.3 Dataset Shapes

For `digits`:

```text
64 -> 64 -> 10
```

For `mnist`:

```text
784 -> 256 -> 10
```

The output size is 10 in both cases because both classify digits 0 through 9.

## 5.4 ANN Training

The ANN is a PyTorch sequential model:

```text
Linear -> ReLU -> Linear
```

The training loop:

- Converts images and labels to tensors.
- Shuffles training examples.
- Runs mini-batches.
- Computes cross-entropy loss.
- Uses Adam to update weights.

At the end, it prints ANN test accuracy.

## 5.5 Quantization

The trained ANN weights are floating-point numbers. RTL uses signed 8-bit
weights.

Quantization maps floating-point values into integers:

```text
float weight -> int8 weight
```

The scale is chosen so the largest absolute weight maps close to 127:

```text
scale = 127 / max(abs(weight))
```

Then each value is rounded and clipped.

## 5.6 Why Quantization Can Change Accuracy

Floating-point values are precise. Int8 values are coarse.

Example:

```text
0.123456 -> maybe 31 after scaling
0.124001 -> maybe 31 after scaling
```

Different float values can become the same integer. This can slightly change
network behavior.

That is why the script reports both:

- ANN accuracy.
- Quantized SNN accuracy.

## 5.7 Rate Pattern Generation

The script creates spike patterns for pixel values.

The idea:

```text
larger pixel value -> more 1s across T_STEPS
smaller pixel value -> fewer 1s across T_STEPS
```

This turns each image into a 3D spike tensor:

```text
image index, timestep, pixel index
```

## 5.8 Golden SNN Simulation

Before exporting to RTL, Python simulates the integer SNN.

This is the golden model.

The golden model matters because it defines what the RTL should reproduce. If
RTL predictions differ from the golden model, the hardware implementation or
testbench has a bug.

## 5.9 Threshold Search

The script searches candidate thresholds:

```text
for threshold in THR_RANGE:
    simulate SNN on tuning subset
    measure accuracy
choose best threshold
```

The chosen threshold is written into `snn_config.vh`.

## 5.10 CSR Export

The function `build_csr` creates source-grouped connectivity.

For each source neuron:

- Pixel input sources connect to hidden neurons.
- Bias source connects to hidden and output neurons.
- Hidden sources connect to output neurons.
- Output neurons have no outgoing synapses.

This produces:

```text
csr_dst
csr_weight
src_start
src_count
```

## 5.11 Memory File Formats

The export function writes:

```text
csr_dst.mem     hex, 16-bit destination IDs
csr_weight.mem  hex, 8-bit signed weights in two's complement
csr_start.mem   hex, 24-bit indices
csr_count.mem   hex, 24-bit counts
snn_spikes.mem  binary, one input spike word per image per timestep
snn_labels.mem  hex labels
snn_config.vh   Verilog macros
```

The width matters because Xcelium reads memory files strictly.

## 5.12 Two's Complement

Signed negative weights are written into hex using two's complement.

For 8-bit values:

```text
-1  -> FF
-2  -> FE
127 -> 7F
```

SystemVerilog reads `csr_weight.mem` into a signed 8-bit memory, so the value is
interpreted as signed again.

## 5.13 Expected Output

A successful MNIST export prints lines like:

```text
ANN test accuracy: ...
Quantized to int8 ...
Best SNN threshold = ...
SNN accuracy on ...
Exported: CSR connectivity ...
Golden accuracy on the 300 exported images: ...
```

Do not submit the RTL server job until export completes.

## 5.14 Common Python Dependency Issues

The import:

```python
import torch
```

requires installing package `torch`, not `pytorch`.

For older Python 3.6 servers, newer PyTorch builds may not work. Use a Python
environment compatible with the required package versions.

The MNIST path also imports TensorFlow Keras:

```python
from tensorflow.keras.datasets import mnist
```

So TensorFlow must be available too.

## 5.15 Exercise

1. Why does the project quantize weights?
2. Why does Python simulate the SNN before RTL?
3. What does threshold search do?
4. Why are CSR files generated instead of storing a full matrix?
5. Why should you check `snn_config.vh` before launching the server run?

