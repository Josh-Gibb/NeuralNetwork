# Neural Network From Scratch (NumPy)

## Overview

This project implements a fully connected feedforward neural network from first principles using NumPy. The primary objective was to develop a rigorous understanding of the underlying mechanics of neural networks by explicitly constructing each component, rather than relying on high-level deep learning frameworks such as PyTorch or TensorFlow.

The implementation includes forward propagation, backpropagation via the chain rule, and parameter optimization using gradient descent.

---

## Features

* Modular `Layer` and `Network` class design
* Multi-layer forward propagation
* Backpropagation with explicit gradient computation
* ReLU activation function and its derivative
* Parameter updates using the Gradient Descent method

---

## Project Structure

```
Layer.py    # Implements individual fully connected layers
Network.py  # Manages network architecture and training logic
Main.py     # Executes training loop and example usage
```

---

## Methodology

### Forward Propagation

For each layer, the network computes a linear transformation followed by a non-linear activation:

* Linear step: `z = W x + b`
* Activation: ReLU applied to hidden layers

This process is repeated sequentially across all layers to produce the final output.

### Backpropagation

Gradients are computed using the chain rule to propagate error signals backward through the network:

* Output error: `dC/da = 2 (y_pred - y_true)`
* Hidden layer errors are calculated using the transpose of the weight matrices and the derivative of the activation function

This enables efficient computation of partial derivatives with respect to each parameter.

### Parameter Updates

Gradients for weights are computed as the outer product of the layer errors and the activations from the previous layer. Bias gradients are derived directly from the error terms.

All parameters are updated using the Gradient Descent method:

* `W := W - η ∂C/∂W`
* `b := b - η ∂C/∂b`

where `η` is the learning rate.

---

## Example

The network is evaluated on a simple regression task:

```python
inputs = [2.0, 3.0]
target = 13.0
hidden_layers = [4, 4]
```

Training produces a decreasing loss over successive epochs:

```
Epoch 1: output=..., loss=...
...
Epoch 100: output ≈ 13, loss ≈ 0
```

---

## Key Takeaways

* Practical implementation of forward and backward propagation
* Explicit derivation and application of gradients
* Understanding how error signals propagate through layered architectures
* Insight into how the Gradient Descent method drives learning in neural networks

---

## Motivation

The purpose of this project was to move beyond black-box usage of machine learning libraries and instead build a concrete understanding of how neural networks operate at a mathematical and computational level.

---

## Future Work

* Implement mini-batch (batch) processing to improve training efficiency and stability
* Evaluate the impact of batch size on convergence speed and generalization
* Compare full-batch vs mini-batch gradient descent in terms of runtime and accuracy

---

## References

* Michael Nielsen, *Neural Networks and Deep Learning* — [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)

---

## Author

Josh Gibb

Recent graduate with a focus on machine learning and computational methods.
