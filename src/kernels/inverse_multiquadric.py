import numpy as np
from typing import Union

from src.kernels.base import BaseKernel
from src.utils.checkers import is_symmetric

class InverseMultiquadricKernel(BaseKernel):
    def __init__(self, lengthscale=1.0,isotropic=True, alpha=1.0, heuristic=False, reference_data=None):
        super().__init__(lengthscale=lengthscale, isotropic=isotropic, heuristic=heuristic)
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
        if self.isotropic:
            # Scalar median heuristic
            pairwise_sq_dists = self._squared_distance_unscaled(x, x)
            upper_tri = pairwise_sq_dists[np.triu_indices_from(pairwise_sq_dists, k=1)]
            return np.sqrt(np.median(upper_tri))

        else:
            # Vector-valued lengthscales, one per dimension
            diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # shape (n, n, d)
            medians = np.median(np.abs(diffs), axis=(0, 1))  # shape (d,)
            return medians + 1e-8  # avoid zero lengthscale


    def compute_grad_x1(self):
        if self._X1 is None or self._X2 is None or self._sq_dist is None:
            raise ValueError("Kernel must be evaluated before calling grad_x1.")

        X1_ = self._X1[:, np.newaxis, :]  # (n1, 1, d)
        X2_ = self._X2[np.newaxis, :, :]  # (1, n2, d)
        diff = X1_ - X2_  # (n1, n2, d)

        if self.isotropic:
            scale2 = self.lengthscale ** 2
            scaled_diff = diff / scale2
        else:
            scale2 = self.lengthscale[np.newaxis, np.newaxis, :] ** 2
            scaled_diff = diff / scale2

        factor = -2 * self.alpha * (1 + self._sq_dist) ** (-self.alpha - 1)
        return factor[..., np.newaxis] * scaled_diff

    def compute_hess_xy(self):
        if self._X1 is None or self._X2 is None or self._sq_dist is None:
            raise ValueError("Kernel must be evaluated with __call__ before calling hess_xy.")

        X1, X2 = self._X1, self._X2
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (n1, n2, d)

        if self.isotropic:
            scale2 = self.lengthscale ** 2

            term1 = (2 * self.alpha / scale2) * (1 + self._sq_dist) ** (-self.alpha - 1)
            term2 = (-4 * self.alpha * (self.alpha + 1) / scale2 ** 2) * \
                    (1 + self._sq_dist) ** (-self.alpha - 2) * np.sum(diff ** 2, axis=-1)

        else:
            scale2 = self.lengthscale[np.newaxis, np.newaxis, :] ** 2
            scaled_diff = diff / scale2

            trace_I = np.sum(1 / self.lengthscale ** 2)  # trace of scaled identity
            term1 = (2 * self.alpha * trace_I) * (1 + self._sq_dist) ** (-self.alpha - 1)
            term2 = (-4 * self.alpha * (self.alpha + 1)) * \
                    (1 + self._sq_dist) ** (-self.alpha - 2) * np.sum(scaled_diff ** 2, axis=-1)

        return term1 - term2  # shape: (n1, n2)