import numpy as np
from typing import Union, Dict

from .base import BaseDistribution


class Laplace(BaseDistribution):
    """
    Univariate Laplace distribution.

    Parameters
    ----------
    mu : float
        Location (mean).
    b : float
        Scale parameter (b > 0).
    """

    def __init__(self, mu: float, b: float):
        assert b > 0, "Scale parameter b must be positive."
        self.mu = mu
        self.b = b
        self.eta = -1.0 / self.b
        self._norm_const = 1.0 / (2.0 * b)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.laplace(self.mu, self.b, size=n_samples).reshape(-1, 1)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        return self._norm_const * np.exp(-np.abs(x - self.mu) / self.b)

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        return -np.log(2 * self.b) - np.abs(x - self.mu) / self.b

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Gradient of log-pdf w.r.t. x.
        For Laplace(mu, b): grad log f(x) = -(sign(x - mu)) / b
        """
        x = np.asarray(x)
        return -np.sign(x - self.mu) / self.b

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of T(x) = |x - mu| wrt x.
        Shape: (..., 1)
        """
        x = np.asarray(x)
        grad = np.sign(x - self.mu)
        return grad[..., None]

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def natural_parameters(self) -> np.ndarray:
        """
        Natural parameterization with fixed mu.
        eta = -1/b (< 0).
        """
        return np.array([self.eta])

    @property
    def parameters_dict(self) -> Dict[str, float]:
        return {"mu": self.mu, "b": self.b}