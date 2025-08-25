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

    def __init__(self, scale: str = "sigma", eps: float = 1e-6):
        assert scale in ("sigma", "log_sigma")
        self.scale = scale
        self.eps = float(eps)

        self._has_data = False
        # raw stats
        self.n = 0
        self.Sx = self.Sy = self.Sxx = self.Sxy = self.Syy = None
        # centered helpers
        self.cx = None            # mean(x)
        self.Sxx_c = None         # sum (x - cx)^2
        self.Sxy_c = None         # sum (x - cx) y

    def set_data(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x, float).reshape(-1)
        y = np.asarray(y, float).reshape(-1)
        if x.shape != y.shape or x.ndim != 1:
            raise ValueError("x and y must be 1D arrays of the same shape.")

        self.n = x.size
        self.Sx = float(x.sum())
        self.Sy = float(y.sum())
        self.Sxx = float(np.dot(x, x))
        self.Sxy = float(np.dot(x, y))
        self.Syy = float(np.dot(y, y))

        # center x for stable T and Q
        self.cx = self.Sx / self.n
        self.Sxx_c = self.Sxx - (self.Sx**2) / self.n       # sum (x - mean)^2
        self.Sxy_c = self.Sxy - self.cx * self.Sy           # sum (x - mean) * y

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

    def _S_Tc_Q(self, alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # S = sum residuals
        S = self.Sy - self.n * alpha - beta * self.Sx
        # T computed stably: T = c*S + (Sxy_c - beta*Sxx_c)
        T_c = self.Sxy_c - beta * self.Sxx_c
        # Q computed with centered x (algebraically identical)
        A = self.Syy - 2 * (alpha + beta * self.cx) * self.Sy + self.n * (alpha + beta * self.cx) ** 2
        Q = A - 2 * beta * self.Sxy_c + beta * beta * self.Sxx_c
        return S, T_c, Q

    def grad_log_pdf_prev(self, theta: ArrayLike, *_ignored) -> np.ndarray:
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
        beta = theta[:, 1]
        gamma = theta[:, 2]

        S, T, Q = self._residual_sums(alpha, beta)

        if self.scale == "sigma":
            sigma = np.maximum(gamma, self.eps)
            sigma2 = sigma**2
            grad_alpha = S / sigma2
            grad_beta = T / sigma2
            grad_gamma = -self.n / sigma + Q / (sigma**3)
        else:
            sigma2 = np.exp(2.0 * gamma)
            grad_alpha = S / sigma2
            grad_beta = T / sigma2
            grad_gamma = -self.n + Q / sigma2

        grad = np.stack([grad_alpha, grad_beta, grad_gamma], axis=-1)
        return grad

    def grad_log_pdf(self, theta: ArrayLike, *_ignored) -> np.ndarray:
        self._ensure_data()
        th = np.asarray(theta, float)
        if th.ndim == 1:
            th = th[None, :]
        if th.shape[1] != 3:
            raise ValueError("theta must have shape (3,) or (m,3): (alpha, beta, gamma).")

        alpha = th[:, 0]
        beta = th[:, 1]
        gamma = th[:, 2]

        S, T_c, Q = self._S_Tc_Q(alpha, beta)

        if self.scale == "sigma":
            sigma = np.maximum(gamma, self.eps)
            sigma2 = sigma * sigma
            grad_alpha = S / sigma2
            grad_beta = (self.cx * S + T_c) / sigma2   # centered form, exact
            grad_gamma = -self.n / sigma + Q / (sigma * sigma2)
        else:
            sigma2 = np.exp(2.0 * gamma)
            grad_alpha = S / sigma2
            grad_beta = (self.cx * S + T_c) / sigma2
            grad_gamma = -self.n + Q / sigma2

        return np.stack([grad_alpha, grad_beta, grad_gamma], axis=-1)


class GaussianARLogLikelihood(BaseLoss):
    """
    AR(K) Gaussian log-likelihood:
        y_t ~ Normal(alpha + sum_{k=1}^K beta_k * y_{t-k}, sigma)
    Parameter vector order: [alpha, beta1, ..., betaK, gamma]
        gamma = sigma        if scale == "sigma"
        gamma = log(sigma)   if scale == "log_sigma"
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self._has_data = False
        self.K = 0
        self.n = 0  # n = T - K

        self.Sy = None                  # sum y_t
        self.Syy = None                 # sum y_t^2
        self.Sx = None                  # sum x_t (K-dim) where x_tk = y_{t-k-1}
        self.Sxx = None                 # sum x_t x_t^T (KxK)
        self.Sxy = None                 # sum x_t y_t (K-dim)

    def set_data(self, y: ArrayLike, K: int) -> None:
        y = np.asarray(y, float).reshape(-1)
        T = y.shape[0]
        if not (1 <= K < T):
            raise ValueError("Require T > K >= 1.")
        self.K = int(K)

        # Targets and lagged design
        Y = y[K:T]  # (n,)
        X = np.column_stack([y[K - k: T - k] for k in range(1, K + 1)])  # (n, K)

        # Sufficient stats over t = K..T-1
        self.n = Y.shape[0]
        self.Sy = float(Y.sum())
        self.Syy = float(Y @ Y)
        self.Sx = X.sum(axis=0)  # (K,)
        self.Sxx = X.T @ X  # (K,K)
        self.Sxy = X.T @ Y  # (K,)
        self._has_data = True

    def _ensure(self):
        if not self._has_data:
            raise RuntimeError("Data not set. Call set_data(y, K).")

    def _S_T_Q(self, alpha: np.ndarray, beta: np.ndarray):
        # alpha: (m,1), beta: (m,K)
        n, Sy, Sx, Sxx, Sxy, Syy = self.n, self.Sy, self.Sx, self.Sxx, self.Sxy, self.Syy

        beta_Sx = beta @ Sx.reshape(-1, 1)  # (m,1)
        S = Sy - n * alpha - beta_Sx  # (m,1)

        T = Sxy[None, :] - alpha * Sx[None, :] - (beta @ Sxx)  # (m,K)

        beta_Sxy = (beta * Sxy[None, :]).sum(axis=1, keepdims=True)  # (m,1)
        beta_Sx = (beta * Sx[None, :]).sum(axis=1, keepdims=True)  # (m,1)
        quad = np.einsum('mi,ij,mj->m', beta, Sxx, beta).reshape(-1, 1)  # (m,1)

        Q = (Syy
             - 2 * alpha * Sy
             - 2 * beta_Sxy
             + n * alpha ** 2
             + 2 * alpha * beta_Sx
             + quad)  # (m,1)
        return S, T, Q

    def grad_log_pdf(self, theta: ArrayLike, *_ignored) -> np.ndarray:
        self._ensure()
        th = np.asarray(theta, float)
        if th.ndim == 1:
            th = th[None, :]
        K = self.K

        alpha = th[:, [0]]  # (m,1)
        beta = th[:, 1:1 + K]  # (m,K)
        gamma = th[:, [1 + K]]  # (m,1)

        S, T, Q = self._S_T_Q(alpha, beta)
        sigma = gamma
        sigma2 = sigma * sigma  # (m,1)
        g_alpha = S / sigma2  # (m,1)
        g_beta = T / sigma2  # (m,K)  (row-wise divide)
        g_gamma = -self.n / sigma + Q / (sigma * sigma2)  # (m,1)

        return np.concatenate([g_alpha, g_beta, g_gamma], axis=1)  # (m, 2+K)
