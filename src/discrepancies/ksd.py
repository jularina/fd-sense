import numpy as np
from typing import Callable
from src.kernels.base import BaseKernel


class KernelizedSteinDiscrepancy:
    """
    Computes the Kernelized Stein Discrepancy (KSD^2) using a V-statistic formulation.

    Parameters
    ----------
    score_fn : Callable[[np.ndarray], np.ndarray]
        Function computing the Stein score vector ∇ log p(x).
        Should return array of shape (m, d).
    kernel : BaseKernel
        Kernel object with precomputed values and gradient methods.
    """

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], np.ndarray],
        kernel: BaseKernel
    ) -> None:
        self.score_fn = score_fn
        self.kernel = kernel

    def compute(self, samples: np.ndarray) -> float:
        """
        Compute the squared Kernelized Stein Discrepancy (KSD^2) using a V-statistic.

        Parameters
        ----------
        samples : np.ndarray
            Posterior samples of shape (m, d).

        Returns
        -------
        float
            The estimated KSD^2 value.
        """
        m, d = samples.shape
        scores = self.score_fn(samples)  # shape (m, d)

        K = self.kernel.value           # (m, m)
        grad1 = self.kernel.grad_x1     # (m, m, d)
        grad2 = self.kernel.grad_x2     # (m, m, d)
        hess = self.kernel.hess_xy      # (m, m) or (m, m, d, d)

        term1 = self._compute_term1(scores, K)
        term2 = self._compute_term2(scores, grad2)
        term3 = self._compute_term3(scores, grad1)
        term4 = self._compute_term4(hess)
        val = np.sum(term1 + term2 + term3 + term4) / (m ** 2)

        if val < 0.0:
            raise ValueError("KSD^2 estimation is negative and failed.")
        else:
            return val

    @staticmethod
    def _compute_term1(scores: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Compute ∑ s(x_i)^T s(x_j) * k(x_i, x_j)"""
        return np.einsum('ik,jk,ij->ij', scores, scores, K)

    @staticmethod
    def _compute_term2(scores: np.ndarray, grad2: np.ndarray) -> np.ndarray:
        """Compute ∑ s(x_i)^T ∇_xj k(x_i, x_j)"""
        return np.einsum('ik,ijk->ij', scores, grad2)

    @staticmethod
    def _compute_term3(scores: np.ndarray, grad1: np.ndarray) -> np.ndarray:
        """Compute ∑ s(x_j)^T ∇_xi k(x_i, x_j)"""
        return np.einsum('jk,ijk->ij', scores, grad1)

    @staticmethod
    def _compute_term4(hess: np.ndarray) -> np.ndarray:
        """
        Compute trace of Hessian ∇²_{xixj} k(x_i, x_j).
        Supports univariate (m, m) or multivariate (m, m, d, d) kernels.
        """
        if hess.ndim == 2:
            return hess
        elif hess.ndim == 4:
            return np.trace(hess, axis1=-2, axis2=-1)
        else:
            raise ValueError(f"Unexpected Hessian shape: {hess.shape}")
