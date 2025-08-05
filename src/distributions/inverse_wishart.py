import numpy as np
from scipy.stats import invwishart
from numpy.linalg import inv, det
from typing import Union

from src.utils.typing import ArrayLike
from .base import BaseDistribution
from src.utils.checkers import is_symmetric_and_psd


class InverseWishart(BaseDistribution):
    """
    Inverse Wishart distribution.

    Parameters
    ----------
    df : float
        Degrees of freedom.
    scale : np.ndarray
        Positive definite scale matrix.
    """

    def __init__(self, df: float, scale: ArrayLike):
        self.df = df
        self.scale = np.asarray(scale)
        self.d = self.scale.shape[0]

        assert self.scale.shape == (self.dim, self.dim), "Scale matrix shape mismatch."
        assert is_symmetric_and_psd(self.scale), "Scale matrix is not p.s.d."

        self.scale_inv = inv(self.scale)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return invwishart.rvs(df=self.df, scale=self.scale, size=n_samples)

    def pdf(self, x: np.ndarray) -> float:
        return invwishart.pdf(x, df=self.df, scale=self.scale)

    def log_pdf(self, x: np.ndarray) -> float:
        return invwishart.logpdf(x, df=self.df, scale=self.scale)

    def grad_log_pdf(self, x: Union[np.ndarray, list]) -> np.ndarray:
        """
        Gradient of log-density of the Inverse Wishart distribution.
        ∇_X log p(X) = ½ X^{-1} Ψ X^{-1} - ½ (ν + d + 1) X^{-1}

        Args:
            x: ndarray of shape (n_samples, d, d), SPD matrices

        Returns:
            grad: ndarray of shape (n_samples, d, d)
        """
        x = np.atleast_3d(x)
        n, d, _ = x.shape
        grad = np.empty_like(x)

        for i in range(n):
            X = x[i]
            X_inv = np.linalg.inv(X)
            term1 = 0.5 * X_inv @ self.psi @ X_inv
            term2 = 0.5 * (self.nu + d + 1) * X_inv
            grad[i] = term1 - term2

        return grad

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of sufficient statistics T(X) = [vec(X^{-1}), log|X|]
        Returns shape (n_samples, d^2, d^2 + 1)
        """
        n, d, _ = x.shape
        d2 = d * d
        jac = np.zeros((n, d2, d2 + 1))

        for i in range(n):
            X = x[i]
            X_inv = np.linalg.inv(X)
            jac[i, :, :d2] = -np.kron(X_inv.T, X_inv)  # ∇ vec(X^{-1})
            jac[i, :, d2] = X_inv.T.reshape(-1)  # ∇ log|X|

        return jac

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        """
        ∇ log h(X) = - (p + 1)/2 * X^{-1}
        """
        x = np.atleast_2d(x)
        return np.zeros_like(x)  # shape (..., d)

    def grad_pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def natural_parameters(self) -> np.ndarray:
        """
        Returns the natural parameter vector η = [η1, η2] for the exponential form:
        """
        eta1 = -0.5 * self.scale.flatten()  # (d^2,)
        eta2 = -(self.df+self.d+1)/2  # (1,)
        return np.concatenate([eta1, eta2])  # (d^2 + 1,)

    def augmented_natural_parameters(self) -> np.ndarray:
        """
        Returns the augmented natural parameter vector [η; 1].
        Shape: (d + d^2 + 1,)
        """
        return np.append(self.natural_parameters(), 1.0)
