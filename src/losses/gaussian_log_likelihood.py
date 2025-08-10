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

    def grad_log_pdf(self, x: ArrayLike, x_bar: float, observations_num: int) -> np.ndarray:
        return observations_num * (x_bar-x) / self.var


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

    def grad_log_pdf(self, x: ArrayLike, x_bar: ArrayLike, observations_num: int) -> np.ndarray:
        """
        Gradient of the log-likelihood w.r.t. parameter x (mean vector).

        Parameters
        ----------
        x : np.ndarray
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
        diff = x_bar - x
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
