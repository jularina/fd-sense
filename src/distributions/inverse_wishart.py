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
        Supports both single (d, d) and batched (num_samples, d, d) inputs.

        Args:
            x (np.ndarray): Covariance matrix or batch of matrices, shape (d, d) or (num_samples, d, d)

        Returns:
            np.ndarray: Gradient matrix or batch, shape (d, d) or (num_samples, d, d)
        """
        if x.ndim == 2:
            # Single matrix case
            x_inv = np.linalg.inv(x)
            grad = -0.5 * ((self.df + self.d + 1) * x_inv - x_inv @ self.scale @ x_inv)
            return grad

        elif x.ndim == 3:
            # Batched case
            num_samples, d1, d2 = x.shape
            assert d1 == self.d and d2 == self.d, "Dimension mismatch in batched Sigma input."

            grads = np.empty_like(x)
            for i in range(num_samples):
                x_inv = np.linalg.inv(x[i])
                grads[i] = -0.5 * ((self.df + self.d + 1) * x_inv - x_inv @ self.scale @ x_inv)
            return grads

        else:
            raise ValueError("Input must be of shape (d, d) or (num_samples, d, d).")

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
