import numpy as np
from typing import Union

from src.kernels.base import BaseKernel
from src.utils.checkers import is_symmetric
from src.utils.typing import ArrayLike


class InverseUnivariateMultiquadricKernel(BaseKernel):
    def __init__(self, lengthscale=1.0, alpha=1.0, heuristic=False, reference_data=None):
        super().__init__(lengthscale=lengthscale, heuristic=heuristic)
        self.alpha = alpha

        if self.heuristic:
            if reference_data is None:
                raise ValueError("reference_data must be provided when heuristic=True")
            self.lengthscale = self._median_heuristic(reference_data)
            print(f"IMQ lenghscale is {self.lengthscale}.")

        # precomputed kernel characteristics
        self._X1 = self._X2 = np.asarray(reference_data)
        self._sq_dist = self._squared_distance(self._X1, self._X2)
        self.value = (1 + self._sq_dist) ** -self.alpha

        if not is_symmetric(self.value):
            raise ValueError("Computed IMQ kernel value is not symmetric.")

        self.grad_x1 = self.compute_grad_x1()
        self.grad_x2 = -self.grad_x1
        self.hess_xy = self.compute_hess_xy()

    def _median_heuristic(self, x: np.ndarray) -> Union[float, np.ndarray]:
        # Scalar median heuristic
        pairwise_sq_dists = self._squared_distance_unscaled(x, x)
        upper_tri = pairwise_sq_dists[np.triu_indices_from(pairwise_sq_dists, k=1)]
        return np.sqrt(np.median(upper_tri))


    def compute_grad_x1(self):
        if self._X1 is None or self._X2 is None or self._sq_dist is None:
            raise ValueError("Kernel must be evaluated before calling grad_x1.")

        X1_ = self._X1[:, np.newaxis, :]  # (n1, 1, d)
        X2_ = self._X2[np.newaxis, :, :]  # (1, n2, d)
        diff = X1_ - X2_  # (n1, n2, d)
        scale2 = self.lengthscale ** 2
        scaled_diff = diff / scale2

        factor = -2 * self.alpha * (1 + self._sq_dist) ** (-self.alpha - 1)
        return factor[..., np.newaxis] * scaled_diff

    def compute_hess_xy(self):
        if self._X1 is None or self._X2 is None or self._sq_dist is None:
            raise ValueError("Kernel must be evaluated with __call__ before calling hess_xy.")

        X1, X2 = self._X1, self._X2
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (n1, n2, d)
        scale2 = self.lengthscale ** 2

        term1 = (2 * self.alpha / scale2) * (1 + self._sq_dist) ** (-self.alpha - 1)
        term2 = (-4 * self.alpha * (self.alpha + 1) / scale2 ** 2) * \
                (1 + self._sq_dist) ** (-self.alpha - 2) * np.sum(diff ** 2, axis=-1)

        return term1 - term2  # shape: (n1, n2)


    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2) / (self.lengthscale ** 2)

        return sq_dist

    def _squared_distance_unscaled(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        return X1_sq + X2_sq.T - 2 * X1 @ X2.T


class InverseMultivariateMultiquadricKernel(BaseKernel):
    def __init__(self, lengthscale: ArrayLike, alpha: float = 1.0,
                 heuristic: bool = False, reference_data: np.ndarray = None):
        super().__init__(lengthscale=lengthscale, heuristic=heuristic)
        self.alpha = alpha
        self.lengthscale = np.asarray(lengthscale)
        if self.heuristic:
            if reference_data is None:
                raise ValueError("reference_data must be provided when heuristic=True")
            self.lengthscale = self._median_heuristic(reference_data)
            print(f"IMQ matrix lengthscale is {self.lengthscale}.")

        # Convert lengthscale to precision matrix (inverse square of lengthscale matrix)
        self.L_inv = self._compute_inverse_scale_matrix(self.lengthscale)

        self._X1 = self._X2 = np.asarray(reference_data)
        self._sq_dist = self._squared_mahalanobis_distance(self._X1, self._X2)
        self.value = (1 + self._sq_dist) ** -self.alpha

        if not is_symmetric(self.value):
            raise ValueError("Computed IMQ kernel value is not symmetric.")

        self.grad_x1 = self.compute_grad_x1()
        self.grad_x2 = -self.grad_x1
        self.hess_xy = self.compute_hess_xy()

    def _compute_inverse_scale_matrix(self, L):
        if np.isscalar(L):
            return np.eye(1) / L**2
        elif L.ndim == 1:
            return np.diag(1.0 / (L ** 2))
        elif L.ndim == 2:
            return np.linalg.inv(L @ L.T)
        else:
            raise ValueError("Lengthscale must be a scalar, 1D array, or 2D matrix.")

    def _median_heuristic(self, x: np.ndarray) -> float:
        pairwise_sq_dists = self._squared_distance_unscaled(x, x)
        upper_tri = pairwise_sq_dists[np.triu_indices_from(pairwise_sq_dists, k=1)]
        return np.sqrt(np.median(upper_tri))

    def _squared_mahalanobis_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # shape: (n1, n2, d)
        sq_dist = np.einsum('nmd,dd,nmd->nm', diff, self.L_inv, diff)
        return sq_dist

    def _squared_distance_unscaled(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)  # shape (n1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)  # shape (n2, 1)
        return X1_sq + X2_sq.T - 2 * X1 @ X2.T

    def compute_grad_x1(self):
        if self._X1 is None or self._X2 is None or self._sq_dist is None:
            raise ValueError("Kernel must be evaluated before calling grad_x1.")

        diff = self._X1[:, np.newaxis, :] - self._X2[np.newaxis, :, :]  # (n1, n2, d)
        factor = -2 * self.alpha * (1 + self._sq_dist) ** (-self.alpha - 1)  # (n1, n2)
        grad = np.einsum('nm,dd,nmd->nmd', factor, self.L_inv, diff)  # (n1, n2, d)
        return grad

    def compute_hess_xy(self):
        if self._X1 is None or self._X2 is None or self._sq_dist is None:
            raise ValueError("Kernel must be evaluated before calling hess_xy.")

        diff = self._X1[:, np.newaxis, :] - self._X2[np.newaxis, :, :]  # (n1, n2, d)
        term1 = 2 * self.alpha * (1 + self._sq_dist) ** (-self.alpha - 1)
        term2 = 4 * self.alpha * (self.alpha + 1) * (1 + self._sq_dist) ** (-self.alpha - 2)

        # Compute Hessian matrix for each pair (i,j): d x d
        outer = np.einsum('nmd,nme->nmde', diff, diff)  # (n1, n2, d, d)
        hess = np.einsum('nm,de->nmde', term1, self.L_inv) - \
               np.einsum('nm,de,nmde->nmde', term2, self.L_inv @ self.L_inv, outer)

        return hess  # (n1, n2, d, d)