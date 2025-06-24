import numpy as np
from abc import ABC, abstractmethod

class BaseKernel(ABC):
    def __init__(self, lengthscale=1.0, variance=1.0, isotropic=True):
        self.variance = variance
        self.lengthscale = lengthscale
        self.isotropic = isotropic

    def _scale_distances(self, X1, X2):
        if self.isotropic:
            scaled_X1 = X1 / self.lengthscale
            scaled_X2 = X2 / self.lengthscale
        else:
            scaled_X1 = X1 / self.lengthscale[np.newaxis, :]
            scaled_X2 = X2 / self.lengthscale[np.newaxis, :]
        return scaled_X1, scaled_X2

    def _squared_distance(self, X1, X2):
        scaled_X1, scaled_X2 = self._scale_distances(X1, X2)
        X1_sq = np.sum(scaled_X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(scaled_X2 ** 2, axis=1).reshape(1, -1)
        return X1_sq + X2_sq - 2 * np.dot(scaled_X1, scaled_X2.T)

    @abstractmethod
    def __call__(self, X1, X2):
        pass