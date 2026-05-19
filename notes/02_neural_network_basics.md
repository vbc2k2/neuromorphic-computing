# Chapter 2: Neural Network Basics

## 2.1 What A Neural Network Does

A neural network maps inputs to outputs.

For handwritten digit classification:

```text
input  -> image pixels
output -> digit class 0 through 9
```

For MNIST, each image is 28 by 28 pixels:

```text
28 * 28 = 784 input values
```

So the network input size is 784.

## 2.2 Layers

This project uses a small multilayer perceptron, or MLP:

```text
input layer -> hidden layer -> output layer
```

For MNIST:

```text
784 -> 256 -> 10
```

That means:

- 784 input pixels.
- 256 hidden neurons.
- 10 output neurons, one for each digit.

## 2.3 Weights

A weight controls how strongly one value affects another.

If input pixel `x` connects to hidden neuron `h`, the contribution is:

```text
x * weight
```

For a hidden neuron, many pixel contributions are added:

```text
sum = x0*w0 + x1*w1 + x2*w2 + ... + bias
```

## 2.4 Bias

A bias is a trainable offset.

Without bias, a neuron can only respond based on weighted inputs. With bias, it
can shift its decision threshold.

In the RTL SNN, the bias is implemented as a special bias neuron that fires
every timestep.

That lets the bias use the same synapse routing mechanism as normal spikes.

## 2.5 Activation Function

After a neuron computes a weighted sum, it usually applies a nonlinear
activation function.

The ANN in `train_snn.py` uses ReLU:

```text
ReLU(x) = max(0, x)
```

ReLU is simple:

- Negative values become 0.
- Positive values pass through.

## 2.6 Output Layer

The output layer has 10 values:

```text
output[0], output[1], ..., output[9]
```

The predicted digit is the index with the largest value:

```text
prediction = argmax(output)
```

If output neuron 7 has the highest value, the network predicts digit 7.

## 2.7 Training

Training means adjusting weights and biases so predictions improve.

The basic loop is:

```text
for each epoch:
    run images through network
    compare predictions to labels
    compute loss
    update weights to reduce loss
```

In this project, PyTorch trains the ANN using:

- Adam optimizer.
- Cross-entropy loss.
- Mini-batches.

## 2.8 Loss

Loss is a number that measures how wrong the network is.

High loss means bad predictions. Low loss means better predictions.

Training tries to reduce loss.

## 2.9 Accuracy

Accuracy is the fraction of correct predictions:

```text
accuracy = correct_predictions / total_images
```

If 98 out of 100 images are correct:

```text
accuracy = 98%
```

The ANN accuracy is useful, but the hardware does not run the floating-point
ANN directly. The project converts the ANN to an integer SNN.

## 2.10 Why Convert To An SNN?

An ANN computes dense numerical layers. An SNN computes sparse spike events.

The conversion is useful because:

- Spikes can be sparse.
- Sparse events allow event-driven hardware.
- Event-driven hardware can skip inactive work.

The tradeoff is that conversion and quantization can reduce accuracy.

## 2.11 Floating Point vs Integer

Training uses floating-point values because they are flexible and precise.

Hardware often prefers integers because they are cheaper:

- Smaller adders.
- Smaller memories.
- Less power.
- Simpler verification.

This project quantizes trained weights into signed 8-bit integers.

## 2.12 Exercise

1. What does the shape `784 -> 256 -> 10` mean?
2. What is a weight?
3. What is a bias?
4. Why does the output layer have 10 neurons?
5. Why does the project not directly run the floating-point ANN in RTL?

