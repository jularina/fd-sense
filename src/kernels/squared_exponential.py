import numpy as np

from src.kernels.base import BaseKernel
from utils.utils import ArrayLike


class SquaredExponentialKernel(BaseKernel):
    """
    Squared Exponential (RBF) kernel, also known as the Gaussian kernel.

    K(x, x') = variance * exp(-0.5 * ||x - x'||^2 / lengthscale^2)
    """

    def __call__(self, X1: ArrayLike, X2: ArrayLike) -> np.ndarray:
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        sq_dist = self._squared_distance(X1, X2)
        return self.variance * np.exp(-0.5 * sq_dist)

    def grad_x1(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Gradient of the kernel wrt X1: ∇₁ k(X1, X2), shape (n1, n2, d)
        """
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        X1_, X2_ = X1[:, np.newaxis, :], X2[np.newaxis, :, :]
        diff = X1_ - X2_  # shape (n1, n2, d)
        sq_dist = np.sum(diff ** 2, axis=-1)  # (n1, n2)
        K = self.variance * np.exp(-0.5 * sq_dist)  # (n1, n2)
        return -K[..., np.newaxis] * diff  # (n1, n2, d)

    def grad_x2(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Gradient of the kernel wrt X2: ∇₂ k(X1, X2), shape (n1, n2, d)
        """
        return -self.grad_x1(X1, X2)

    def hess_xy(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Trace of the Hessian ∇₁∇₂ k(X1, X2), shape (n1, n2)
        """
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        n1, d = X1.shape
        X1_, X2_ = X1[:, np.newaxis, :], X2[np.newaxis, :, :]
        diff = X1_ - X2_  # (n1, n2, d)
        sq_dist = np.sum(diff ** 2, axis=-1)  # (n1, n2)
        K = self.variance * np.exp(-0.5 * sq_dist)  # (n1, n2)
        trace_hess = K * (np.sum(diff ** 2, axis=-1) - d)  # scalar trace of matrix
        return trace_hess