from typing import Tuple

import numpy as np

from src.utils.typing import ArrayLike
from src.losses.base import BaseLoss


class GaussianLogLikelihood(BaseLoss):
    """
    Univariate Gaussian distribution.
    """

    def __init__(self, mu: float, sigma: float):
        assert sigma > 0, "Standard deviation must be positive."
        self.mu = mu
        self.sigma = sigma
        self.var = sigma ** 2
        self._norm_const = 1.0 / np.sqrt(2 * np.pi * self.var)

    def grad_log_pdf(self, theta: ArrayLike, x_bar: float, observations_num: int) -> np.ndarray:
        return observations_num * (x_bar-theta) / self.var


class MultivariateGaussianLogLikelihood(BaseLoss):
    """
    Multivariate Gaussian log-likelihood with full covariance.
    """

    def __init__(self, mu: ArrayLike, cov: ArrayLike):
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)

        assert self.cov.shape[0] == self.cov.shape[1], "Covariance must be square."
        assert self.cov.shape[0] == self.mu.shape[0], "Covariance and mean dimension mismatch."

        if not np.all(np.linalg.eigvals(self.cov) > 0):
            raise ValueError("Covariance matrix must be positive definite.")

        self.dim = self.mu.shape[0]
        self.cov_inv = np.linalg.inv(self.cov)
        self.det_cov = np.linalg.det(self.cov)
        self._norm_const = 1.0 / np.sqrt((2 * np.pi) ** self.dim * self.det_cov)

    def grad_log_pdf(self, theta: ArrayLike, x_bar: ArrayLike, observations_num: int) -> np.ndarray:
        """
        Gradient of the log-likelihood w.r.t. parameter x (mean vector).

        Parameters
        ----------
        theta : np.ndarray
            Current parameter (mean vector), shape (d,)
        x_bar : np.ndarray
            Empirical mean of the data, shape (d,)
        observations_num : int
            Number of data points

        Returns
        -------
        grad : np.ndarray
            Gradient vector of shape (d,)
        """
        diff = x_bar - theta
        result = diff @ self.cov_inv.T

        return observations_num * result

    def grad_log_pdf_wrt_cov(self, Sigma: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """
        Gradient of the log-likelihood w.r.t. the covariance matrix Sigma,
        assuming mean mu is known.

        Parameters
        ----------
        Sigma : np.ndarray
            Covariance matrix parameter (d, d)
        observations : np.ndarray
            Observed data, shape (n, d)

        Returns
        -------
        grad : np.ndarray
            Gradient of log-likelihood w.r.t. Sigma, shape (d, d)
        """
        n, d = observations.shape
        centered = observations - self.mu
        S = centered.T @ centered
        Sigma_inv = np.linalg.inv(Sigma)
        grad = 0.5 * (Sigma_inv @ S @ Sigma_inv - n * Sigma_inv)

        return grad.reshape(Sigma.shape[0], -1)


class GaussianLinearRegressionLogLikelihood(BaseLoss):
    """
    Log-likelihood for y ~ Normal(alpha + beta * x, sigma).

    Initialize with scale/eps only; provide data later via set_data(x, y).

    Parameters
    ----------
    scale : {"sigma", "log_sigma"}
        If "sigma", the third parameter is sigma (must be > 0).
        If "log_sigma", the third parameter is gamma = log(sigma) (unconstrained).
    eps : float
        Numerical floor to keep sigma away from 0 when scale == "sigma".
    """

    def __init__(self, scale: str = "sigma", eps: float = 1e-12):
        assert scale in ("sigma", "log_sigma")
        self.scale = scale
        self.eps = float(eps)

        # Data-dependent sufficient statistics (set after set_data)
        self._has_data = False
        self.n = 0
        self.Sx = self.Sy = self.Sxx = self.Sxy = self.Syy = None

    def set_data(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        Assign observations and precompute sufficient statistics.
        x, y must be 1D arrays of equal length.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.shape != y.shape or x.ndim != 1:
            raise ValueError("x and y must be 1D arrays of the same shape.")

        self.n = x.size
        self.Sx = float(x.sum())
        self.Sy = float(y.sum())
        self.Sxx = float(np.dot(x, x))
        self.Sxy = float(np.dot(x, y))
        self.Syy = float(np.dot(y, y))
        self._has_data = True

    def _ensure_data(self):
        if not self._has_data:
            raise RuntimeError("Data not set. Call set_data(x, y) before grad_log_pdf.")

    def _residual_sums(self, alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        S = self.Sy - self.n * alpha - beta * self.Sx
        T = self.Sxy - alpha * self.Sx - beta * self.Sxx
        Q = (self.Syy
             - 2 * alpha * self.Sy
             - 2 * beta * self.Sxy
             + self.n * alpha**2
             + 2 * alpha * beta * self.Sx
             + beta**2 * self.Sxx)
        return S, T, Q

    def grad_log_pdf(self, theta: ArrayLike, *_ignored) -> np.ndarray:
        """
        Gradient of log-likelihood w.r.t. (alpha, beta, gamma).
        Supports theta shape (3,) or (m, 3). Returns same leading shape.

        If scale == "sigma": gamma = sigma (>0)
            d/d alpha = S / sigma^2
            d/d beta  = T / sigma^2
            d/d sigma = -n/sigma + Q/sigma^3

        If scale == "log_sigma": gamma = log(sigma)
            d/d alpha = S / sigma^2
            d/d beta  = T / sigma^2
            d/d log_sigma = -n + Q/sigma^2
        """
        self._ensure_data()
        alpha = theta[:, 0]
        beta  = theta[:, 1]
        gamma = theta[:, 2]

        S, T, Q = self._residual_sums(alpha, beta)

        if self.scale == "sigma":
            sigma = np.maximum(gamma, self.eps)
            sigma2 = sigma**2
            grad_alpha = S / sigma2
            grad_beta  = T / sigma2
            grad_gamma = -self.n / sigma + Q / (sigma**3)
        else:
            sigma2 = np.exp(2.0 * gamma)
            grad_alpha = S / sigma2
            grad_beta  = T / sigma2
            grad_gamma = -self.n + Q / sigma2

        grad = np.stack([grad_alpha, grad_beta, grad_gamma], axis=-1)
        return grad
