from typing import Callable

ParametricFloat = Callable[[int], float]


def linear_annealing(start: float, end: float, duration: int) -> ParametricFloat:
    """
    Factory that creates a parametric function that linearly anneals between
    :start: and :end: for :duration: time steps.
    """

    def wrapped(t: int):
        """CODE HERE:
        return the epsilon for each time step within the duration.
        if time > duration, use the end value, else use the scheduled value

        Let t_hat = min(duration, t)
        Let delta = end - start
        Let f(t) = start + (t_hat / duration) * delta.

        t = 0         => t_hat = 0        => f(t) = start
        t < duration  => t_hat = t        => f(t) = start + (t / duration) delta
        t >= duration => t_hat = duration => f(t) = end

        :param t:
        :return:
        """
        t_hat = min(duration, t)
        return start + (t_hat / duration) * (end - start)

    return wrapped


def constant(c: float) -> ParametricFloat:
    """
    Factory that creates a parametric function that linearly anneals between
    :start: and :end: for :duration: time steps.
    """

    def wrapped(_: int):
        """CODE HERE:
        return the epsilon for each time step within the duration.
        if time > duration, use the end value, else use the scheduled value

        Let t_hat = min(duration, t)
        Let delta = end - start
        Let f(t) = start + (t_hat / duration) * delta.

        t = 0         => t_hat = 0        => f(t) = start
        t < duration  => t_hat = t        => f(t) = start + (t / duration) delta
        t >= duration => t_hat = duration => f(t) = end

        :param _:
        :return:
        """
        return c

    return wrapped