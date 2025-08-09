import numpy as np

from src.kernels.base import BaseKernel
from src.utils.checkers import is_symmetric
from src.utils.typing import ArrayLike


class InverseUnivariateMultiquadricKernel(BaseKernel):
    def __init__(self, lengthscale=1.0, alpha=1.0, heuristic=False, reference_data=None):
        super().__init__(lengthscale=lengthscale, heuristic=heuristic)
        self.alpha = alpha
        self._X1 = self._X2 = np.asarray(reference_data)
        self._sq_dist = self._squared_distance(self._X1, self._X2)
        self.value = (1 + self._sq_dist) ** -self.alpha

        if not is_symmetric(self.value):
            raise ValueError("Computed IMQ kernel value is not symmetric.")

        self.grad_x1 = self.compute_grad_x1()
        self.grad_x2 = np.swapaxes(self.grad_x1, 0, 1)
        self.hess_xy = self.compute_hess_xy()

    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        return np.sum(diff ** 2, axis=2) / (self.lengthscale ** 2)

    def compute_grad_x1(self) -> np.ndarray:
        """
        Gradient of the kernel with respect to x1.
        """
        if self._sq_dist is None:
            raise ValueError("Kernel must be evaluated before calling grad_x1.")

        X1_, X2_ = self._X1[:, np.newaxis, :], self._X2[np.newaxis, :, :]
        diff = X1_ - X2_  # (n1, n2, d)
        scale2 = self.lengthscale ** 2
        scaled_diff = diff / scale2
        factor = -2 * self.alpha * (1 + self._sq_dist) ** (-self.alpha - 1)

        return factor[..., np.newaxis] * scaled_diff

    def compute_hess_xy(self):
        """
        Hessian of the kernel with respect to x and y.
        """
        if self._sq_dist is None:
            raise ValueError("Kernel must be evaluated before calling hess_xy.")

        diff = self._X1[:, np.newaxis, :] - self._X2[np.newaxis, :, :]
        scale2 = self.lengthscale ** 2
        term1 = (2 * self.alpha / scale2) * (1 + self._sq_dist) ** (-self.alpha - 1)
        term2 = (4 * self.alpha * (self.alpha + 1) / scale2 ** 2) * \
                (1 + self._sq_dist) ** (-self.alpha - 2) * np.sum(diff ** 2, axis=-1)

        return term1 - term2  # shape: (n1, n2)


class InverseMultivariateMultiquadricKernel(BaseKernel):
    """
    Inverse multiquadric kernel with matrix (Mahalanobis) lengthscale.
    K(x, x') = (1 + (x - x')^T L^{-1} (x - x'))^(-alpha)
    """

    def __init__(self, lengthscale: ArrayLike, alpha: float = 1.0, heuristic: bool = False, reference_data: np.ndarray = None):
        super().__init__(lengthscale=lengthscale, heuristic=heuristic, reference_data=reference_data)
        self.alpha = alpha
        self.lengthscale = np.asarray(self.lengthscale)
        self.L_inv = self._compute_inverse_scale_matrix()

        self._X1 = self._X2 = np.asarray(reference_data)
        self._sq_dist = self._squared_distance(self._X1, self._X2)
        self.value = (1 + self._sq_dist) ** -self.alpha

        if not is_symmetric(self.value):
            raise ValueError("Computed IMQ kernel value is not symmetric.")

        self.grad_x1 = self.compute_grad_x1()
        self.grad_x2 = -self.grad_x1
        self.hess_xy = self.compute_hess_xy()

    def _compute_inverse_scale_matrix(self) -> np.ndarray:
        if np.isscalar(self.lengthscale):
            return np.eye(1) / self.lengthscale**2
        elif self.lengthscale.ndim == 1:
            return np.diag(1.0 / (self.lengthscale ** 2))
        elif self.lengthscale.ndim == 2:
            return np.linalg.inv(self.lengthscale @ self.lengthscale.T)
        else:
            raise ValueError("Lengthscale must be a scalar, 1D array, or 2D matrix.")

    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # shape: (n1, n2, d)

        return np.einsum('nmd,dd,nmd->nm', diff, self.L_inv, diff)

    def compute_grad_x1(self) -> np.ndarray:
        if self._sq_dist is None:
            raise ValueError("Kernel must be evaluated before calling grad_x1.")

        diff = self._X1[:, np.newaxis, :] - self._X2[np.newaxis, :, :]  # (n1, n2, d)
        factor = -2 * self.alpha * (1 + self._sq_dist) ** (-self.alpha - 1)  # (n1, n2)

        return np.einsum('nm,dd,nmd->nmd', factor, self.L_inv, diff)  # (n1, n2, d)

    def compute_hess_xy(self) -> np.ndarray:
        if self._sq_dist is None:
            raise ValueError("Kernel must be evaluated before calling hess_xy.")

        diff = self._X1[:, np.newaxis, :] - self._X2[np.newaxis, :, :]  # (n1, n2, d)
        term1 = 2 * self.alpha * (1 + self._sq_dist) ** (-self.alpha - 1)
        term2 = 4 * self.alpha * (self.alpha + 1) * (1 + self._sq_dist) ** (-self.alpha - 2)
        outer = np.einsum('nmd,nme->nmde', diff, diff)  # (n1, n2, d, d)
        hess = np.einsum('nm,de->nmde', term1, self.L_inv) - \
            np.einsum('nm,de,nmde->nmde', term2, self.L_inv @ self.L_inv, outer)  # (n1, n2, d, d)

        return hess
