"""
Represents a layer in a fully connected neural network from scratch.

Handles:
- Forward pass using ReLU
- Backpropagation error computation
- Weight updates
- Bias updates
- Gradient descent
"""

import numpy as np
from typing import *

learning_rate = 1e-3

def create_weights(x, y):
    """
    Creates a weight matrix with small random values.

    :param x: Number of rows
    :param y: Number of columns
    :return: Weight matrix with values in [0, 0.1)
    """
    return np.random.rand(x, y) * 0.1

def create_biases(x):
    """
    Creates a bias vector initialized to zeros.

    :param x: Size of the vector
    :return: A vector of biases
    """
    return np.zeros(x)

def relu(z):
    """
    Applies the ReLU activation function elementwise.

    :param z: Input data (scalar, vector, or matrix)
    :return: Array with ReLU applied to each element
    """
    return np.maximum(0, z)

def relu_prime(z):
    """
    Applies the derivative of ReLU elementwise.

    :param z: Input data (scalar, vector, or matrix)
    :return: Array where positive values become 1 and non-positive values become 0
    """
    return np.where(z > 0, 1, 0)

class layer:
    """
    Represents a layer in a fully connected neural network.
    """
    def __init__(self, num_of_neurons: int):
        """
        Initializes a layer with a given number of neurons.

        :param num_of_neurons: Number of neurons in the layer
        """
        self.num_of_neurons = num_of_neurons
        self.activations = None
        self.biases = None
        self.weights = None
        self.weight_grads = None
        self.errors = None
        self.z_score = None

    def first_layer(self, inputs: List['float'], num_of_neurons: int):
        """
        Sets the activations for the input layer.

        :param inputs: Input data vector
        :param num_of_neurons: Number of neurons in the layer
        """
        self.activations = np.array(inputs)

    def create_layer(self, prev_layer: "layer", num_of_neurons: int):
        """
        Initializes weights and biases for this layer and computes activations.

        :param prev_layer: Previous layer
        :param num_of_neurons: Number of neurons in this layer
        """
        self.weights = create_weights(num_of_neurons, prev_layer.num_of_neurons)
        self.biases = create_biases(num_of_neurons)
        self.create_activations(prev_layer.activations)

    def create_activations(self, prev_acts, output=False):
        """
        Computes the layer's activations from the previous layer.

        :param prev_acts: Activations from the previous layer
        :param output: If True, uses the raw z values without ReLU
        """
        self.z_score = self.weights @ prev_acts + self.biases
        if output:
            self.activations = self.z_score
        else:
            self.activations = relu(self.z_score)

    def last_level_errors(self, dc_da):
        """
        Stores the output-layer error term.

        :param dc_da: Derivative of cost with respect to activation
        """
        self.errors = dc_da

    def calc_errors(self, next_layer: "layer"):
        """
        Computes the error term for this layer using the next layer.

        :param next_layer: The next layer in the network
        """
        delta = np.ravel(next_layer.errors)
        self.errors = (next_layer.weights.T @ delta) * relu_prime(self.z_score)

    def update_weights(self, prev_layer: "layer"):
        """
        Updates the weights using gradient descent.

        :param prev_layer: The previous layer
        """
        prev_acts = np.ravel(prev_layer.activations)
        errs = np.ravel(self.errors)
        self.weight_grads = np.outer(errs, prev_acts)
        self.weights -= self.weight_grads * learning_rate

    def update_biases(self):
        """
        Updates the biases using gradient descent.
        """
        self.biases -= self.errors * learning_rate