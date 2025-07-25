import numpy as np
from scipy.stats import invwishart
from .base import BaseDistribution

from src.utils.typing import ArrayLike


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

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return invwishart.rvs(df=self.df, scale=self.scale, size=n_samples)

    def pdf(self, x: np.ndarray) -> float:
        return invwishart.pdf(x, df=self.df, scale=self.scale)

    def log_pdf(self, x: np.ndarray) -> float:
        return invwishart.logpdf(x, df=self.df, scale=self.scale)

    def grad_log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of log Inverse Wishart density with respect to x (Sigma).

        Args:
            x (np.ndarray): The covariance matrix Sigma (d x d)

        Returns:
            np.ndarray: Gradient matrix (d x d)
        """
        x_inv = np.linalg.inv(x)
        grad = -0.5 * ((self.df + self.d + 1) * x_inv - x_inv @ self.scale @ x_inv)

        return grad

    def grad_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of sufficient statistics T(X) = [X^{-1}, log|X|]
        Returns an array of shape (d, d+1) flattened
        """
        X_inv = np.linalg.inv(x)
        log_det = np.log(np.linalg.det(x))
        return np.concatenate([X_inv.flatten(), [log_det]])

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        """
        ∇ log h(X) = - (p + 1)/2 * X^{-1}
        """
        return -0.5 * (self.d + 1) * np.linalg.inv(x)

    def grad_pdf(self, x: np.ndarray) -> np.ndarray:
        pass
