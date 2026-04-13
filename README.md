# Neural Network From Scratch (NumPy)

## Overview

This project is a fully connected neural network implemented from scratch using only NumPy. The goal was to understand how neural networks actually work under the hood by building each component manually instead of relying on high-level frameworks like PyTorch or TensorFlow.

The implementation includes forward propagation, backpropagation, gradient descent, and a simple training loop.

---

## Features

* Custom `Layer` and `Network` classes
* Forward propagation across multiple layers
* Backpropagation using the chain rule
* ReLU activation function and its derivative
* Gradient descent for updating weights and biases
* Simple regression example for testing

---

## Project Structure

```
Layer.py    # Defines a single neural network layer
Network.py  # Handles forward + backward propagation across layers
Main.py     # Runs training loop and example
```

---

## How It Works

### Forward Pass

Each layer performs:

* Linear transformation: `z = W * x + b`
* Activation (ReLU for hidden layers)

### Backward Pass

* Compute output error: `dC/da = 2 * (prediction - target)`
* Propagate errors backward through layers using:

  * Transposed weights
  * Activation derivatives

### Weight Updates

* Gradients are computed using the outer product of errors and previous activations
* Parameters are updated using gradient descent

---

## Example

The model is trained on a simple regression task:

```python
inputs = [2.0, 3.0]
target = 13.0
hidden_layers = [4, 4]
```

Training output:

```
Epoch 1: output=..., loss=...
...
Epoch 100: output≈13, loss≈0
```

---

## What I Learned

* How forward and backward propagation work mathematically
* How gradients are computed and applied
* How neural networks learn from data step-by-step
* The importance of activation functions in learning non-linear relationships

---

## Next Steps

* Implement mini-batching
* Add different activation functions (Sigmoid, Tanh)
* Introduce more advanced optimizers (Adam, Momentum)
* Extend to classification problems

---

## Why This Project

This project was built to move beyond using machine learning libraries and develop a deeper understanding of the underlying mechanics of neural networks.

---

## Author

Built as part of a personal deep learning journey.
