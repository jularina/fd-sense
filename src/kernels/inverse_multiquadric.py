import numpy as np

from src.kernels.base import BaseKernel


class InverseMultiquadricKernel(BaseKernel):
    def __init__(self, lengthscale=1.0, variance=1.0, isotropic=True, alpha=1.0):
        super().__init__(lengthscale, variance, isotropic)
        self.alpha = alpha

    def __call__(self, X1, X2):
        X1, X2 = np.asarray(X1), np.asarray(X2)
        sq_dist = self._squared_distance(X1, X2)
        return self.variance * (1 + sq_dist) ** -self.alpha

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