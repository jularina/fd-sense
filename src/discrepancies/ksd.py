import numpy as np
from typing import Callable
from src.kernels.base import BaseKernel


class KernelizedSteinDiscrepancy:
    """
    Compute KSD using V-statistic formulation.

    Parameters
    ----------
    score_fn : Callable[[np.ndarray], np.ndarray]
        Function computing the score vector ∇ log p(x).
    kernel : BaseKernel
        Kernel object with gradient and Hessian methods.
    """

    def __init__(self, score_fn: Callable[[np.ndarray], np.ndarray], kernel: BaseKernel):
        self.score_fn = score_fn
        self.kernel = kernel

    def compute(self, samples: np.ndarray) -> float:
        """
        Compute the KSD^2 using V-statistic for a set of samples.

        Parameters
        ----------
        samples : np.ndarray
            Samples of shape (m, d)

        Returns
        -------
        float
            The KSD^2 value.
        """
        m, d = samples.shape
        scores = self.score_fn(samples)  # (m, d)

        K = self.kernel.value          # (m, m)
        grad1 = self.kernel.grad_x1        # (m, m, d)
        grad2 = self.kernel.grad_x2        # (m, m, d)
        hess = self.kernel.hess_xy         # (m, m)

        term1 = np.einsum('ik,jk,ij->ij', scores, scores, K)
        term2 = np.einsum('ik,ijk->ij', scores, grad2)
        term3 = np.einsum('jk,ijk->ij', scores, grad1)
        term4 = hess

        total = term1 + term2 + term3 + term4
        return np.sum(total) / (m * m)
