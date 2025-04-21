import numpy as np


class FirstOrderOde:
    def evaluate(self, t: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Evaluates the first-order ODE dy/dt = f(t, y).

        Parameters:
        t (np.ndarray[float]): The independent variable (time).
        y (np.ndarray[float]): The dependent variable (state).

        Returns:
        np.ndarray[float]: The rate of change of the dependent variable.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def step_euler(
        self, t: np.ndarray[float], y: np.ndarray[float], dt: float
    ) -> np.ndarray[float]:
        """
        Performs a single Euler step to update the state.

        Parameters:
        t (np.ndarray[float]): The independent variable (time).
        y (np.ndarray[float]): The dependent variable (state).
        dt (float): The time step for the Euler method.

        Returns:
        np.ndarray[float]: The updated state after the Euler step.
        """
        return y + self.evaluate(t, y) * dt


class LogisticGrowth(FirstOrderOde):
    def __init__(self, r: float, K: float):
        """
        Initializes the logistic growth model.

        Parameters:
        r (float): The intrinsic growth rate.
        K (float): The carrying capacity.
        """
        self.r = r
        self.K = K

    def evaluate(self, t: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Evaluates the logistic growth model.

        Parameters:
        t (np.ndarray[float]): The independent variable (time).
        y (np.ndarray[float]): The dependent variable (population).

        Returns:
        np.ndarray[float]: The rate of change of the population.
        """
        return self.r * y * (1 - y / self.K)
