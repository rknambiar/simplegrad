import random
from sklearn.datasets import make_moons

from layer import MLP
from utils import draw_contour_plot

BATCH_SIZE = 100
NUM_EPOCS = 150

def main():
    dataset = make_moons(noise=0.1, random_state=60)
    x, y = dataset
    y = [1 if yi == 1 else -1 for yi in y] # convert labels to +1, -1

    network = MLP()
    network.add_layer(2, 16)
    network.add_layer(16, 16)
    network.add_layer(16, 16)
    network.add_layer(16, 1, activation="linear")
    print(network)

    for epoch in range(NUM_EPOCS):
        x_batch, y_batch = zip(*random.sample(list(zip(x, y)), BATCH_SIZE))
        
        # Forward pass
        y_pred = list(map(network, x_batch))

        # hinge loss
        loss = [(1 - yi_batch * yi_pred[0]).relu() for yi_pred, yi_batch in zip(y_pred, y_batch)]
        loss = sum(loss) * (1. / len(loss))

        # compute accuracy
        accuracy = [(yi_pred[0].value > 0) == (yi_batch > 0) for yi_pred, yi_batch in zip(y_pred, y_batch)]
        accuracy = sum(accuracy)
        
        # set gradients to zero
        network.zero_grad()

        # Compute gradients
        loss.backward()

        learning_rate = 0.03
        for weight in network.parameters():
            weight.value -= learning_rate * weight.grad

        draw_contour_plot(network, x, y, epoch)

        print(f"Epoch: {epoch}, loss: {loss}, accuracy: {accuracy}")

if __name__ == '__main__':
    main()