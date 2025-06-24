import numpy as np
from typing import Union
from .base import BaseDistribution

class Exponential(BaseDistribution):
    """
    Exponential distribution.

    Parameters
    ----------
    rate : float
        Rate parameter (lambda).
    """

    def __init__(self, rate: float):
        assert rate > 0, "Rate must be positive."
        self.rate = rate

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.exponential(1.0 / self.rate, size=n_samples)

    def pdf(self, x: Union[float, np.ndarray]) -> float:
        x = np.asarray(x)
        pdf_val = self.rate * np.exp(-self.rate * x)
        pdf_val[x < 0] = 0  # PDF is zero for x < 0
        return pdf_val

    def log_pdf(self, x: Union[float, np.ndarray]) -> float:
        x = np.asarray(x)
        log_val = np.log(self.rate) - self.rate * x
        log_val[x < 0] = -np.inf
        return log_val

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        grad = -self.rate * np.ones_like(x)
        grad[x < 0] = 0
        return grad

    def grad_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return self.grad_log_pdf(x) * self.pdf(x)
