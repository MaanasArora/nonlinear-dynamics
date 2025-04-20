import numpy as np
from matplotlib import pyplot as plt
from .ode import FirstOrderOde


def plot_solve_ode(ode: FirstOrderOde, t0: float, t1: float, y0: float, dt: float):
    """
    Solves and plots the first-order ODE using Euler's method.

    Parameters:
    ode (FirstOrderOde): The first-order ODE to solve.
    t0 (float): The initial time.
    t1 (float): The final time.
    y0 (float): The initial state.
    dt (float): The time step for the numerical method.
    """
    t_values = np.arange(t0, t1, dt)
    y_values = np.zeros(len(t_values))

    for i, t in enumerate(t_values):
        y_values[i] = y0
        y0 = ode.step_euler(t, y0, dt)

    plt.plot(t_values, y_values, label=f"{ode.__class__.__name__} (Euler)")
    plt.xlabel("t")
    plt.ylabel("y")
