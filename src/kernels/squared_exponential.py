import numpy as np

from src.kernels.base import BaseKernel


class SquaredExponentialKernel(BaseKernel):
    def __call__(self, X1, X2):
        sq_dist = self._squared_distance(X1, X2)
        return self.variance * np.exp(-0.5 * sq_dist)