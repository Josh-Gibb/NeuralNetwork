import numpy as np
from typing import *

learning_rate = 1e-4

def create_weights(x, y):
    return np.random.rand(x, y) * 0.1

def create_biases(x):
    return np.zeros(x)

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return np.where(z > 0, 1, 0)

class layer:
    def __init__(self, num_of_neurons: int):
        self.num_of_neurons = num_of_neurons
        self.activations = None
        self.biases = None
        self.weights = None
        self.weight_grads = None
        self.errors = None
        self.z_score = None

    def first_layer(self, inputs: List['float'], num_of_neurons: int):
        self.activations = np.array(inputs)

    def create_layer(self, prev_layer: "layer", num_of_neurons: int):
        self.weights = create_weights(num_of_neurons, prev_layer.num_of_neurons)
        self.biases = create_biases(num_of_neurons)
        self.create_activations(prev_layer.activations)

    def create_activations(self, prev_acts, output = False):
        self.z_score = self.weights @ prev_acts + self.biases
        if output:
            self.activations = self.z_score
        else:
            self.activations = relu(self.z_score)

    def last_level_errors(self, dc_da):
        self.errors = dc_da

    def calc_errors(self, next_layer: "layer"):
        delta = np.ravel(next_layer.errors)
        self.errors = (next_layer.weights.T @ delta) * relu_prime(self.z_score)

    def update_weights(self, prev_layer: "layer"):
        prev_acts = np.ravel(prev_layer.activations)
        errs = np.ravel(self.errors)
        self.weight_grads =np.outer(errs, prev_acts)
        self.weights -= self.weight_grads * learning_rate

    def update_biases(self):
        self.biases -= self.errors * learning_rate

