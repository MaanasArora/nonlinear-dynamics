class FirstOrderOde:
    def evaluate(self, t: float, y: float) -> float:
        """
        Evaluates the first-order ODE dy/dt = f(t, y).

        Parameters:
        t (float): The independent variable (time).
        y (float): The dependent variable (state).

        Returns:
        float: The derivative of y with respect to t.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def step_euler(self, t: float, y: float, dt: float) -> float:
        """
        Performs a single Euler step to update the state.

        Parameters:
        t (float): The current time.
        y (float): The current state.
        dt (float): The time step.

        Returns:
        float: The updated state after the Euler step.
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

    def evaluate(self, t: float, y: float) -> float:
        """
        Evaluates the logistic growth model.

        Parameters:
        t (float): The independent variable (time).
        y (float): The dependent variable (population size).

        Returns:
        float: The rate of change of the population size.
        """
        return self.r * y * (1 - y / self.K)
