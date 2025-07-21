import numpy as np
from abc import ABC
from typing import Union
from src.utils.typing import ArrayLike


class BaseKernel(ABC):
    def __init__(self, lengthscale: Union[float, ArrayLike] = 1.0, heuristic: bool = False,):
        self.lengthscale = lengthscale
        self.heuristic = heuristic
        self.value = None
        self.grad_x1 = None
        self.grad_x2 = None
        self.hess_xy = None

    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        pass

    def _squared_distance_unscaled(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        pass