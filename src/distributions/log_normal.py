import numpy as np
from typing import Union
from .base import BaseDistribution
from typing import Dict


class LogNormal(BaseDistribution):
    """
    Univariate Log-Normal distribution.

    Parameters
    ----------
    mu_log : float
        Mean  = exp(mu_log)
    sigma_log : float
    """

    def __init__(self, mu_log: float, sigma_log: float):
        assert sigma_log > 0, "Standard deviation must be positive."
        self.mu = mu_log
        self.sigma = sigma_log
        self.var = self.sigma ** 2
        self._norm_const = 1.0 / (np.sqrt(2 * np.pi * self.var))

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.lognormal(self.mu, self.sigma, size=n_samples)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        pdf = np.zeros_like(x, dtype=np.float64)
        mask = x > 0
        x_pos = x[mask]
        pdf[mask] = (self._norm_const / x_pos) * np.exp(-((np.log(x_pos) - self.mu) ** 2) / (2 * self.var))
        return pdf

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        log_pdf = np.full_like(x, -np.inf, dtype=np.float64)
        mask = x > 0
        x_pos = x[mask]
        log_pdf[mask] = (
            -np.log(x_pos)
            - 0.5 * np.log(2 * np.pi * self.var)
            - ((np.log(x_pos) - self.mu) ** 2) / (2 * self.var)
        )
        return log_pdf

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        grad = np.zeros_like(x, dtype=np.float64)
        mask = x > 0
        x_pos = x[mask]
        grad[mask] = -1 / x_pos - (np.log(x_pos) - self.mu) / (self.var * x_pos)
        return grad

    def natural_parameters(self) -> np.ndarray:
        eta1 = self.mu / self.var
        eta2 = -0.5 / self.var
        return np.array([eta1, eta2])

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        grad = np.zeros((x.shape[0], 2))
        mask = x > 0
        grad[mask, 0] = 1 / x[mask]
        grad[mask, 1] = 2 * np.log(x[mask]) / x[mask]
        return grad[:, None, :]

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        grad = np.zeros_like(x, dtype=np.float64)
        mask = x > 0
        grad[mask] = -1 / x[mask]
        return grad

    @property
    def parameters_dict(self) -> Dict[str, float]:
        return {"mu": self.mu, "sigma": self.sigma}