import numpy as np
from abc import ABC, abstractmethod
from typing import Union

from src.utils.typing import ArrayLike

class BaseKernel(ABC):
    def __init__(self, lengthscale: Union[float, np.ndarray] = 1.0, variance: float = None, isotropic: bool = True, heuristic: bool = False,):
        self.lengthscale = lengthscale
        self.variance = variance
        self.isotropic = isotropic
        self.heuristic = heuristic
        self.value = None
        self.grad_x1 = None
        self.grad_x2 = None
        self.hess_xy = None

    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.isotropic:
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2) / (self.lengthscale ** 2)
        else:
            diff = (X1[:, np.newaxis, :] - X2[np.newaxis, :, :])  # shape (n1, n2, d)
            scaled_diff = diff / self.lengthscale[np.newaxis, np.newaxis, :]
            sq_dist = np.sum(scaled_diff ** 2, axis=2)

        return sq_dist

    def _squared_distance_unscaled(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        return X1_sq + X2_sq.T - 2 * X1 @ X2.T