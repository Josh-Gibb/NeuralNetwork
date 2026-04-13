"""
Neural network implementation

Handles:
- Creating layers based on a given structure
- Forward pass through the network
- Backpropagation to calculate errors
- Updating weights and biases using gradient descent
"""
from Layer import layer
from typing import List

class network:
    def __init__(self, inputs: List['float'], target: float, neurons_per_layer: List[int]):
        """
        Creates a neural network.

        :param inputs: Input values for the network
        :param target: Expected output value
        :param neurons_per_layer: Number of neurons in each hidden layer
        """
        neurons_per_layer.append(1)
        neurons_per_layer.insert(0, len(inputs))
        self.neurons_per_layer = neurons_per_layer
        self.layers = [layer(neuron) for neuron in neurons_per_layer]
        self.inputs = inputs
        self.target = target

    def initialize_network(self):
        """
        Initializes the network layers.
        """
        self.layers[0].first_layer(self.inputs, self.neurons_per_layer[1])
        for i in range(1, len(self.neurons_per_layer)):
            self.layers[i].create_layer(self.layers[i - 1], self.neurons_per_layer[i])

    def forward(self):
        """
        Runs a forward pass through the network.
        """
        for i in range(len(self.layers) - 1):
            output = (i == len(self.layers) - 2)
            self.layers[i + 1].create_activations(self.layers[i].activations, output)

    def backward(self):
        """
        Runs backpropagation and updates weights and biases.
        """
        for i in range(len(self.layers) - 1, 0, -1):
            if i == len(self.layers) - 1:
                self.layers[i].last_level_errors(2 * (self.output() - self.target))
            else:
                self.layers[i].calc_errors(self.layers[i + 1])
            self.layers[i].update_weights(self.layers[i - 1])
            self.layers[i].update_biases()

    def output(self) -> float:
        """
        Returns the network output.
        """
        return self.layers[-1].activations[0]

    def debug_state(self, epoch=None):
        """
        Prints the current state of the network.

        :param epoch: Current epoch number
        """
        print("\n" + "-" * 40)
        if epoch is not None:
            print(f"Epoch {epoch}")
        print("-" * 40)

        for i, layer in enumerate(self.layers):
            print(f"\nLayer {i}")

            if hasattr(layer, "activations"):
                print("activations:")
                print(layer.activations)

            if hasattr(layer, "weights"):
                print("weights:")
                print(layer.weights)

            if hasattr(layer, "biases"):
                print("biases:")
                print(layer.biases)

            if hasattr(layer, "weight_grads"):
                print("weight_grads:")
                print(layer.weight_grads)

        print("-" * 40 + "\n")