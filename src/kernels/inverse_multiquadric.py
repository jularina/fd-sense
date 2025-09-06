import numpy as np

from src.kernels.base import BaseKernel
from src.utils.checkers import is_symmetric
from src.utils.typing import ArrayLike


class InverseUnivariateMultiquadricKernel(BaseKernel):
    def __init__(self, lengthscale=1.0, alpha=1.0, heuristic=False, reference_data=None):
        super().__init__(lengthscale=lengthscale, heuristic=heuristic, reference_data=reference_data)
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
    IMQ kernel: K(x, y) = (1 + r^2)^(-alpha),  r^2 = (x - y)^T M (x - y),  M = L^{-1}

    ∇_x K = -2 alpha (1 + r^2)^(-alpha-1) M (x - y)
    ∇_y K =  2 alpha (1 + r^2)^(-alpha-1) M (x - y)  = -∇_x K

    ∇_x ∇_y K = 2 alpha (1 + r^2)^(-alpha-1) M
                - 4 alpha (alpha + 1) (1 + r^2)^(-alpha-2) [M(x - y)][M(x - y)]^T

    trace(∇_x ∇_y K) = 2 alpha (1 + r^2)^(-alpha-1) tr(M)
                        - 4 alpha (alpha + 1) (1 + r^2)^(-alpha-2) ||M(x - y)||^2
    """

    def __init__(self, lengthscale: ArrayLike, alpha: float = 1.0,
                 heuristic: bool = False, reference_data: np.ndarray = None,
                 compute_full_hessian: bool = False):
        super().__init__(lengthscale=lengthscale, heuristic=False, reference_data=reference_data)
        if reference_data is None:
            raise ValueError("reference_data must be provided")
        self.alpha = float(alpha)

        X = np.asarray(reference_data, dtype=np.float64)
        self._X1 = self._X2 = X
        n, d = X.shape

        if heuristic:
            self.lengthscale = self._median_heuristic_per_dim(X)
        else:
            self.lengthscale = np.asarray(lengthscale, dtype=np.float64)

        self.M = self._compute_inverse_scale_matrix(d)

        self.X1M = self._X1 @ self.M
        self.X2M = self._X2 @ self.M

        self._sq_dist = self._squared_distance(self._X1, self._X2, self.X1M, self.X2M)
        self.value = (1.0 + self._sq_dist) ** (-self.alpha)

        if not is_symmetric(self.value):
            raise ValueError("Computed IMQ kernel value is not symmetric.")

        self.grad_x1 = self.compute_grad_x1()
        self.grad_x2 = np.swapaxes(self.grad_x1, 0, 1)

        # always present (n, n), works with your ndim check
        self.hess_xy = self.compute_hess_xy_trace()

        # optional full tensor if you need it
        self.hess_xy_full = self.compute_hess_xy_full() if compute_full_hessian else None

    def _compute_inverse_scale_matrix(self, d: int) -> np.ndarray:
        ls = self.lengthscale
        if np.ndim(ls) == 0:
            return np.eye(d, dtype=np.float64) / (float(ls) ** 2)
        if np.ndim(ls) == 1:
            return np.diag(1.0 / (np.asarray(ls, dtype=np.float64) ** 2))
        if np.ndim(ls) == 2:
            return np.asarray(np.linalg.inv(ls), dtype=np.float64)
        raise ValueError("Lengthscale must be scalar, 1D array, or 2D SPD matrix.")

    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray,
                          X1M: np.ndarray, X2M: np.ndarray) -> np.ndarray:
        s1 = np.einsum("ij,ij->i", X1, X1M)
        s2 = np.einsum("ij,ij->i", X2, X2M)
        cross = X1M @ X2.T
        return s1[:, None] + s2[None, :] - 2.0 * cross

    def compute_grad_x1(self) -> np.ndarray:
        factor = -2.0 * self.alpha * (1.0 + self._sq_dist) ** (-self.alpha - 1.0)
        Mdiff = self.X1M[:, None, :] - self.X2M[None, :, :]
        return factor[..., None] * Mdiff

    def compute_hess_xy_trace(self) -> np.ndarray:
        term1 = 2.0 * self.alpha * (1.0 + self._sq_dist) ** (-self.alpha - 1.0)
        term2 = 4.0 * self.alpha * (self.alpha + 1.0) * (1.0 + self._sq_dist) ** (-self.alpha - 2.0)
        Mdiff = self.X1M[:, None, :] - self.X2M[None, :, :]
        norm_Mdiff_sq = np.sum(Mdiff * Mdiff, axis=-1)
        return term1 * np.trace(self.M) - term2 * norm_Mdiff_sq

    def compute_hess_xy_full(self) -> np.ndarray:
        term1 = 2.0 * self.alpha * (1.0 + self._sq_dist) ** (-self.alpha - 1.0)
        term2 = 4.0 * self.alpha * (self.alpha + 1.0) * (1.0 + self._sq_dist) ** (-self.alpha - 2.0)
        Mdiff = self.X1M[:, None, :] - self.X2M[None, :, :]
        return term1[..., None, None] * self.M - term2[..., None, None] * (Mdiff[..., :, None] * Mdiff[..., None, :])
