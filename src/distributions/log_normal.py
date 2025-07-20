import numpy as np
from typing import Union
from .base import BaseDistribution


class LogNormal(BaseDistribution):
    """
    Univariate Log-Normal distribution.

    Parameters
    ----------
    mu : float
        Mean of the underlying normal distribution.
    sigma : float
        Standard deviation of the underlying normal distribution.
    """

    def __init__(self, mu: float, sigma: float):
        assert sigma > 0, "Standard deviation must be positive."
        self.mu = mu
        self.sigma = sigma
        self.var = sigma ** 2
        self._norm_const = 1.0 / (np.sqrt(2 * np.pi * self.var))

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.lognormal(self.mu, self.sigma, size=n_samples)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        pdf = np.zeros_like(x, dtype=np.float64)
        positive = x > 0
        x_pos = x[positive]
        pdf[positive] = (self._norm_const / x_pos) * np.exp(-((np.log(x_pos) - self.mu) ** 2) / (2 * self.var))
        return pdf

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        log_pdf = np.full_like(x, -np.inf, dtype=np.float64)
        positive = x > 0
        x_pos = x[positive]
        log_pdf[positive] = (
            -np.log(x_pos)
            - 0.5 * np.log(2 * np.pi * self.var)
            - ((np.log(x_pos) - self.mu) ** 2) / (2 * self.var)
        )
        return log_pdf

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        grad = np.zeros_like(x, dtype=np.float64)
        positive = x > 0
        x_pos = x[positive]
        grad[positive] = -1 / x_pos - (np.log(x_pos) - self.mu) / (self.var * x_pos)
        return grad

    def grad_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return self.grad_log_pdf(x) * self.pdf(x)