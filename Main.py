from Network import network

if __name__ == '__main__':
    inputs = [5, 7, 8, 6, 3, -5, 5, 9, 0, 2]
    hidden_layers = [3, 3, 3, 3]
    target = 50

    net = network(inputs, target, hidden_layers)
    net.initialize_network()

    for epoch in range(10000):
        net.forward()
        y = net.output()
        loss = (y - target) ** 2
        print(f"Epoch {epoch + 1}: output={y}, loss={loss}")
        net.backward()
