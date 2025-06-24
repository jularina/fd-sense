import numpy as np
from numpy.linalg import inv, det
from typing import Union
from .base import BaseDistribution


class Gaussian(BaseDistribution):
    """
    Univariate Gaussian distribution.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    """

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        self.var = sigma ** 2
        self._norm_const = 1.0 / (np.sqrt(2 * np.pi * self.var))

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.normal(self.mu, self.sigma, size=n_samples)

    def pdf(self, x: Union[float, np.ndarray]) -> float:
        return self._norm_const * np.exp(-((x - self.mu) ** 2) / (2 * self.var))

    def grad_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return self.grad_log_pdf(x) * self.pdf(x)

    def log_pdf(self, x: Union[float, np.ndarray]) -> float:
        return -0.5 * np.log(2 * np.pi * self.var) - ((x - self.mu) ** 2) / (2 * self.var)

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return -(x - self.mu) / self.var


class MultivariateGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution.

    Parameters
    ----------
    mu : np.ndarray
        Mean vector.
    cov : np.ndarray
        Covariance matrix.
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray):
        self.mu = mu
        self.cov = cov
        self.cov_inv = inv(cov)
        self.dim = len(mu)
        self._norm_const = 1.0 / np.sqrt((2 * np.pi) ** self.dim * det(cov))

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.multivariate_normal(self.mu, self.cov, size=n_samples)

    def pdf(self, x: np.ndarray) -> float:
        dx = x - self.mu
        return self._norm_const * np.exp(-0.5 * dx.T @ self.cov_inv @ dx)

    def grad_pdf(self, x: np.ndarray) -> np.ndarray:
        return self.grad_log_pdf(x) * self.pdf(x)

    def log_pdf(self, x: np.ndarray) -> float:
        dx = x - self.mu
        return np.log(self._norm_const) - 0.5 * dx.T @ self.cov_inv @ dx

    def grad_log_pdf(self, x: np.ndarray) -> np.ndarray:
        return -self.cov_inv @ (x - self.mu)