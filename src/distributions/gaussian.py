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
        assert sigma > 0, "Standard deviation must be positive."
        self.mu = mu
        self.sigma = sigma
        self.var = sigma ** 2
        self._norm_const = 1.0 / np.sqrt(2 * np.pi * self.var)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.normal(self.mu, self.sigma, size=n_samples)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        return self._norm_const * np.exp(-((x - self.mu) ** 2) / (2 * self.var))

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asarray(x)
        return -0.5 * np.log(2 * np.pi * self.var) - ((x - self.mu) ** 2) / (2 * self.var)

    def grad_log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return -(x - self.mu) / self.var

    def grad_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return self.grad_log_pdf(x) * self.pdf(x)


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
        self.dim = self.mu.shape[0]
        assert self.cov.shape == (self.dim, self.dim), "Covariance shape mismatch."

        self.cov_inv = inv(self.cov)
        self._norm_const = 1.0 / np.sqrt((2 * np.pi) ** self.dim * det(self.cov))

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return np.random.multivariate_normal(self.mu, self.cov, size=n_samples)

    def pdf(self, x: Union[np.ndarray, list]) -> np.ndarray:
        x = np.atleast_2d(x)
        dx = x - self.mu
        exponent = np.einsum("...i,ij,...j->...", dx, self.cov_inv, dx)
        return self._norm_const * np.exp(-0.5 * exponent)

    def log_pdf(self, x: Union[np.ndarray, list]) -> np.ndarray:
        x = np.atleast_2d(x)
        dx = x - self.mu
        exponent = np.einsum("...i,ij,...j->...", dx, self.cov_inv, dx)
        return np.log(self._norm_const) - 0.5 * exponent

    def grad_log_pdf(self, x: Union[np.ndarray, list]) -> np.ndarray:
        x = np.atleast_2d(x)
        dx = x - self.mu
        return -np.einsum("ij,...j->...i", self.cov_inv, dx)

    def grad_pdf(self, x: Union[np.ndarray, list]) -> np.ndarray:
        return self.grad_log_pdf(x) * self.pdf(x)[..., np.newaxis]