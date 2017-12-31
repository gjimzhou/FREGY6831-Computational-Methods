# PlotFunction.py

"""
Wheels for plot functions
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3D(option_prices, stock_prices, times, z_label, num):
    fig = plt.figure(num)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(stock_prices, times, option_prices, rstride=2, cstride=2, cmap=plt.cm.coolwarm,
        linewidth=0.5, antialiased=True)

    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time')
    ax.set_zlabel(z_label)
    ax.grid(True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot2D(option_prices, stock_prices, y_label):
    plt.plot(stock_prices, option_prices)
    plt.xlabel('stock price')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()
