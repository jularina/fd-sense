import numpy as np
from abc import ABC, abstractmethod
from typing import Union

from src.utils.typing import ArrayLike

class BaseKernel(ABC):
    def __init__(self, lengthscale: Union[float, np.ndarray] = 1.0, variance: float = 1.0, isotropic: bool = True, heuristic: bool = False,):
        self.lengthscale = lengthscale
        self.variance = variance
        self.isotropic = isotropic
        self.heuristic = heuristic

    def _scale_distances(self, X1: np.ndarray, X2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.isotropic:
            scaled_X1 = X1 / self.lengthscale
            scaled_X2 = X2 / self.lengthscale
        else:
            scaled_X1 = X1 / self.lengthscale[np.newaxis, :]
            scaled_X2 = X2 / self.lengthscale[np.newaxis, :]
        return scaled_X1, scaled_X2

    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        scaled_X1, scaled_X2 = self._scale_distances(X1, X2)
        X1_sq = np.sum(scaled_X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(scaled_X2 ** 2, axis=1, keepdims=True)
        return X1_sq + X2_sq.T - 2 * scaled_X1 @ scaled_X2.T

    @abstractmethod
    def __call__(self, X1: ArrayLike, X2: ArrayLike) -> np.ndarray:
        pass

    @abstractmethod
    def grad_x1(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """∇₁ k(X1, X2) — gradient wrt first argument"""
        pass

    @abstractmethod
    def grad_x2(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """∇₂ k(X1, X2) — gradient wrt second argument"""
        pass

    @abstractmethod
    def hess_xy(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """trace of ∇₁∇₂ k(X1, X2)"""
        pass