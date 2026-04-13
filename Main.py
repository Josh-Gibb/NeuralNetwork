"""
Runs and trains the neural network.

Handles:
- Setting up input data, target, and network structure
- Initializing the network
- Running the training loop (forward + backward pass)
"""
from Network import network

if __name__ == '__main__':
    inputs = [2.0, 3.0]
    target = 13.0
    hidden_layers = [4, 4]

    net = network(inputs, target, hidden_layers)
    net.initialize_network()

    for epoch in range(100):
        net.forward()
        y = net.output()
        loss = (y - target) ** 2
        print(f"Epoch {epoch + 1}: output={y}, loss={loss}")
        net.backward()
