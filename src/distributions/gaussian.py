import numpy as np
from numpy.linalg import inv, det
from typing import Union
from .base import BaseDistribution

from src.utils.typing import ArrayLike


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

    def sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """
        T(x): Sufficient statistics for the Gaussian.
        For univariate Gaussian: T(x) = [x, x^2]
        Returns shape (..., 2)
        """
        x = np.asarray(x)
        return np.stack([x[..., 0], x[..., 0] ** 2], axis=-1)

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """
        ∇T(x): Jacobian of sufficient statistics.
        For univariate Gaussian: [1, 2x]
        Returns shape (..., 1, 2)
        """
        x = np.asarray(x)
        grad_T1 = np.ones_like(x[..., 0])
        grad_T2 = 2 * x[..., 0]
        grad = np.stack([grad_T1, grad_T2], axis=-1)
        return grad[..., None, :]  # shape (..., 1, 2)

    def base_measure(self, x: np.ndarray) -> np.ndarray:
        """
        h(x): Base measure of the Gaussian distribution.
        For standard Gaussian in exponential form, h(x) = 1.
        """
        return np.ones_like(x[..., 0])

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        """
        ∇ log h(x): Gradient of log base measure.
        For Gaussian, log h(x) = 0 → ∇ log h(x) = 0
        """
        return np.zeros_like(x)  # shape (..., 1)

    def natural_parameters(self) -> np.ndarray:
        """
        Returns the natural parameter vector η for the exponential form.
        For univariate Gaussian:
        η = [mu / sigma^2, -1 / (2 * sigma^2)]
        """
        eta1 = self.mu / self.var
        eta2 = -0.5 / self.var
        return np.array([eta1, eta2])

    def augmented_natural_parameters(self) -> np.ndarray:
        """
        Returns the augmented natural parameter vector: [η; 1]
        """
        return np.append(self.natural_parameters(), 1.0)


class MultivariateGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution.

    Parameters
    ----------
    mu : list[float]
        Mean vector.
    cov : list[list[float]]
        Covariance matrix.
    """

    def __init__(self, mu: ArrayLike, cov: ArrayLike):
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)
        self.dim = self.mu.shape[0]
        # assert self.cov.shape == (self.dim, self.dim), "Covariance shape mismatch."

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
        return - np.einsum("ij,...j->...i", self.cov_inv, dx)

    def grad_pdf(self, x: Union[np.ndarray, list]) -> np.ndarray:
        return self.grad_log_pdf(x) * self.pdf(x)[..., np.newaxis]

    def natural_parameters(self) -> np.ndarray:
        """
        Returns the natural parameter vector η = [η1, η2] for the exponential form:
        η1 = Σ^{-1} μ (shape: d,)
        η2 = -0.5 * vec(Σ^{-1}) (shape: d^2,)
        Result shape: (d + d^2,)
        """
        eta1 = self.cov_inv @ self.mu  # (d,)
        eta2 = -0.5 * self.cov_inv.flatten()  # vec(Sigma^{-1}) → (d^2,)
        return np.concatenate([eta1, eta2])  # (d + d^2,)

    def augmented_natural_parameters(self) -> np.ndarray:
        """
        Returns the augmented natural parameter vector [η; 1].
        Shape: (d + d^2 + 1,)
        """
        return np.append(self.natural_parameters(), 1.0)