import numpy as np
from abc import ABC
from typing import Union
from src.utils.typing import ArrayLike


class BaseKernel(ABC):
    """
    Abstract base class for kernel functions.
    Handles lengthscale, heuristic initialization, and placeholder methods for distance computation.
    """

    def __init__(self, lengthscale: Union[float, ArrayLike] = 1.0, heuristic: bool = False, reference_data: np.ndarray = None):
        self.heuristic = heuristic
        self.lengthscale = lengthscale

        if self.heuristic:
            if reference_data is None:
                raise ValueError("reference_data must be provided when heuristic=True")
            self.lengthscale = self._median_heuristic(reference_data)

        self._X1: Union[None, np.ndarray] = None
        self._X2: Union[None, np.ndarray] = None
        self._sq_dist: Union[None, np.ndarray] = None
        self.value: Union[None, np.ndarray] = None
        self.grad_x1: Union[None, np.ndarray] = None
        self.grad_x2: Union[None, np.ndarray] = None
        self.hess_xy: Union[None, np.ndarray] = None

    def _squared_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Placeholder for scaled squared distance (to be implemented in subclasses).
        """
        raise NotImplementedError

    def _squared_distance_unscaled(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Computes the unscaled squared Euclidean distance between two sets of points.
        """
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        return X1_sq + X2_sq.T - 2 * X1 @ X2.T

    def _median_heuristic(self, x: np.ndarray) -> float:
        """
        Computes the median heuristic bandwidth (sqrt of median pairwise squared distance).
        """
        pairwise_sq_dists = self._squared_distance_unscaled(x, x)
        upper_tri = pairwise_sq_dists[np.triu_indices_from(pairwise_sq_dists, k=1)]
        return np.sqrt(np.median(upper_tri))

    def _median_heuristic_per_dim(self, x: np.ndarray, *, jitter: float = 1e-12) -> np.ndarray:
        """
        Per-dimension median heuristic.
        Returns a vector of lengthscales l = sqrt(median_{i<j} (x_i - x_j)^2) per dimension.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("reference_data must be a 2D array of shape (n, d).")
        n, d = x.shape
        if n < 2:
            return np.sqrt(np.var(x, axis=0) + jitter)

        diffs = x[:, None, :] - x[None, :, :]
        iu = np.triu_indices(n, k=1)
        diffs = diffs[iu]

        med_sq = np.median(diffs**2, axis=0)
        return np.sqrt(med_sq + jitter)
