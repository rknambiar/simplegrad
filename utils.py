import matplotlib.pyplot as plt
import numpy as np
import os

def draw_contour_plot(network, train_data, train_label, epoch, save=False):
    x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
    y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
    
    x = np.arange(x_min, x_max, 0.2)
    y = np.arange(y_min, y_max, 0.2)
    xx, yy = np.meshgrid(x, y)

    inputs = np.vstack((xx.ravel(), yy.ravel())).T
    scores = list(map(network, inputs))
    scores = np.array([s[0].value > 0 for s in scores])

    scores = scores.reshape(xx.shape)

    plt.clf()
    plt.contourf(xx, yy, scores, levels=1, alpha=0.5, cmap='coolwarm')
    plt.scatter(train_data[:, 0], train_data[:, 1], 
                c=train_label, s=40, cmap='coolwarm')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.pause(0.5)

    if save:
        path = os.path.join('./images', str(epoch) + '.png')
        plt.savefig(path)