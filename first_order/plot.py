import numpy as np
from matplotlib import pyplot as plt
from .ode import FirstOrderOde


def plot_field_ode(
    ode: FirstOrderOde, x0: float, x1: float, y0: float, y1: float, num_points: int = 20
):
    """
    Plots the vector field of a first-order ODE.

    Parameters:
    ode (FirstOrderOde): The first-order ODE to plot.
    x0 (float): The lower bound of the x-axis.
    x1 (float): The upper bound of the x-axis.
    y0 (float): The lower bound of the y-axis.
    y1 (float): The upper bound of the y-axis.
    num_points (int): The number of points in each direction for the grid.
    """
    step = (x1 - x0) / num_points

    x = np.linspace(x0, x1, num_points)
    y = np.linspace(y0, y1, num_points)
    X, Y = np.meshgrid(x, y)

    U = np.ones_like(X)
    V = ode.evaluate(X, Y) * step

    plt.quiver(X, Y, U, V)
