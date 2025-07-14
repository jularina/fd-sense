import numpy as np
from typing import Union

from src.kernels.base import BaseKernel


class InverseMultiquadricKernel(BaseKernel):
    def __init__(self, lengthscale=1.0, variance=1.0, isotropic=True, alpha=1.0, heuristic=False, reference_data=None):
        super().__init__(lengthscale, variance, isotropic, heuristic)
        self.alpha = alpha

        if self.heuristic:
            if reference_data is None:
                raise ValueError("reference_data must be provided when heuristic=True")
            self.lengthscale = self._median_heuristic(reference_data)

    def __call__(self, X1, X2):
        X1, X2 = np.asarray(X1), np.asarray(X2)
        sq_dist = self._squared_distance(X1, X2)
        return self.variance * (1 + sq_dist) ** -self.alpha

    def _median_heuristic(self, X: np.ndarray) -> Union[float, np.ndarray]:
        if self.isotropic:
            # Scalar median heuristic
            pairwise_sq_dists = self._squared_distance(X, X)
            upper_tri = pairwise_sq_dists[np.triu_indices_from(pairwise_sq_dists, k=1)]
            return np.sqrt(np.median(upper_tri))

        else:
            # Vector-valued lengthscales, one per dimension
            diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # shape (n, n, d)
            medians = np.median(np.abs(diffs), axis=(0, 1))  # shape (d,)
            return medians + 1e-8  # avoid zero lengthscale

    def grad_x1(self, X1, X2):
        X1, X2 = np.asarray(X1), np.asarray(X2)
        n1, d = X1.shape
        n2 = X2.shape[0]
        X1_ = X1[:, np.newaxis, :]  # (n1, 1, d)
        X2_ = X2[np.newaxis, :, :]  # (1, n2, d)
        diff = X1_ - X2_  # (n1, n2, d)
        sq_dist = np.sum(diff ** 2, axis=-1)  # (n1, n2)
        factor = -2 * self.alpha * self.variance * (1 + sq_dist) ** (-self.alpha - 1)
        return factor[..., np.newaxis] * diff  # (n1, n2, d)

    def grad_x2(self, X1, X2):
        return -self.grad_x1(X1, X2)

    def hess_xy(self, X1, X2):
        X1, X2 = np.asarray(X1), np.asarray(X2)
        X1_ = X1[:, np.newaxis, :]
        X2_ = X2[np.newaxis, :, :]
        diff = X1_ - X2_
        sq_dist = np.sum(diff ** 2, axis=-1)  # (n1, n2)
        base = (1 + sq_dist) ** (-self.alpha - 2)
        term1 = 2 * self.alpha * self.variance * (1 + sq_dist) ** (-self.alpha - 1) * X1.shape[1]  # trace of I
        term2 = 4 * self.alpha * (self.alpha + 1) * self.variance * base * np.sum(diff ** 2, axis=-1)
        return term1 - term2  # shape: (n1, n2)