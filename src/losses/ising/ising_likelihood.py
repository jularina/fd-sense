import numpy as np

from src.utils.typing import ArrayLike
from src.losses.base import BaseLoss


# class IsingLikelihood(BaseLoss):
#     """
#     Ising Model Likelihood (ignoring normalization constant).
#     4-neighbour l x l grid, where d = l*l.
#     """
#
#     def __init__(self, theta: float):
#         assert theta > 0, "Theta must be positive."
#         self.theta = theta
#         self._A = None
#         self._d = None
#
#     @staticmethod
#     def _grid_adjacency(d: int) -> np.ndarray:
#         """Build symmetric zero-diagonal adjacency for an l x l grid (4-neighbour)."""
#         l = int(np.sqrt(d))
#         A = np.zeros((d, d), dtype=float)
#
#         # map (r, c) -> idx = r*l + c
#         for r in range(l):
#             for c in range(l):
#                 i = r * l + c
#                 # up
#                 if r > 0:
#                     j = (r - 1) * l + c
#                     A[i, j] = A[j, i] = 1.0
#                 # down
#                 if r < l - 1:
#                     j = (r + 1) * l + c
#                     A[i, j] = A[j, i] = 1.0
#                 # left
#                 if c > 0:
#                     j = r * l + (c - 1)
#                     A[i, j] = A[j, i] = 1.0
#                 # right
#                 if c < l - 1:
#                     j = r * l + (c + 1)
#                     A[i, j] = A[j, i] = 1.0
#         return A
#
#     def _get_adjacency(self, d: int) -> np.ndarray:
#         self._A = self._grid_adjacency(d)
#         self._d = d
#
#     def grad_log_pdf(self, theta: ArrayLike, x: ArrayLike) -> np.ndarray:
#         """
#         Gradient of the (unnormalized) log-likelihood wrt theta, summed over all datapoints.
#
#         Parameters
#         ----------
#         theta : (M, 1) array-like of positive values
#         x     : (N, d) array-like (binary {0,1}); d must equal l*l
#
#         Returns
#         -------
#         grad : (M, 1) ndarray
#         """
#         theta = np.asarray(theta, dtype=float).reshape(-1, 1)   # (M,1)
#         x = np.asarray(x, dtype=float)                          # (N,d)
#         assert x.ndim == 2, "x must be (N, d)."
#         assert np.all(theta > 0), "All theta must be positive."
#
#         d = x.shape[1]
#         self._get_adjacency(d)
#         stats = np.einsum("nd,df,nf->n", x, self._A, x)   # (N,)
#         S_total = float(np.sum(stats))
#         grad = -(S_total / (theta ** 2))
#
#         return grad


class IsingLikelihoodGivenGrads(BaseLoss):
    """
    Ising Model Likelihood (ignoring normalization constant).
    4-neighbour l x l grid, where d = l*l.
    """

    def __init__(self):
        self.grad_log_likelihood = None

    def grad_log_pdf(self) -> np.ndarray:
        return self.grad_log_likelihood