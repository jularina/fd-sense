import numpy as np

from src.kernels.base import BaseKernel
from src.discrepancies.ksd import KernelizedSteinDiscrepancy
from src.bayesian_model.base import BayesianModel
from src.utils.typing import ArrayLike


class PosteriorKSD:
    def __init__(
        self,
        samples: ArrayLike,
        model: BayesianModel,
        kernel: BaseKernel,
    ):
        """
        Computes Kernelized Stein Discrepancy (KSD) components between posterior samples and candidate posterior.

        Args:
            samples: Posterior samples (m, D)
            model: Bayesian model with score functions
            kernel: Kernel object with derivative methods
        """
        self.samples = samples
        self.model = model
        self.kernel = kernel
        self.ksd = KernelizedSteinDiscrepancy(model.posterior_score, kernel)

    def estimate_ksd(self) -> float:
        return self.ksd.compute(self.samples)

    def compute_ksd_quadratic_form(self):
        JT_aug_T = self._compute_augmented_jacobians()  # (m, p+1, d)
        K, grad1, grad2, trace_hess = self._compute_kernel_quantities()
        Lambda_m = self._compute_Lambda(JT_aug_T, K)
        b_prior = self._compute_b_prior(JT_aug_T, grad1, grad2)
        b_cross = self._compute_b_cross(JT_aug_T, K)
        b_m = b_prior + b_cross
        C = self._compute_C(trace_hess)
        return Lambda_m, b_m, C

    def _compute_augmented_jacobians(self):
        J_T = self.model.jacobian_sufficient_statistics(self.samples)  # (m, d, p)
        s_h = self.model.grad_log_base_measure(self.samples)  # (m, d)
        JT_aug = np.concatenate([J_T, s_h[..., None]], axis=2)  # (m, d, p+1)
        JT_aug_T = np.transpose(JT_aug, (0, 2, 1))  # (m, p+1, d)
        return JT_aug_T

    def _compute_kernel_quantities(self):
        K = self.kernel.value  # (m, m)
        grad1 = self.kernel.grad_x1  # (m, m, d)
        grad2 = self.kernel.grad_x2  # (m, m, d)
        hess = self.kernel.hess_xy  # (m, m, d, d) or (m, m)

        if hess.ndim == 4:
            trace_hess = np.trace(hess, axis1=-2, axis2=-1)  # (m, m)
        elif hess.ndim == 2:
            trace_hess = hess
        else:
            raise ValueError(f"Unexpected hessian shape: {hess.shape}")

        return K, grad1, grad2, trace_hess

    def _compute_Lambda(self, JT_aug_T: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Efficient computation of:
            Lambda = (1 / m^2) * sum_{i,j} JT_aug_T[i] @ JT_aug_T[j].T * K[i, j]

        JT_aug_T: shape (m, p1, d)
        K: shape (m, m)
        Returns:
            Lambda: shape (p1, p1)
        """
        m, p1, d = JT_aug_T.shape

        # If d = 1, we can squeeze to (m, p1)
        if d == 1:
            J = JT_aug_T.squeeze(-1)  # shape (m, p1)
            JT_outer = J[:, None, :] * J[None, :, :]  # shape (m, m, p1)
            # Use einsum to compute outer products with kernel weighting
            Lambda = np.einsum("ij,ijp,ijq->pq", K, JT_outer, JT_outer) / (m ** 2)
        else:
            # For d > 1, more general case
            Lambda = np.zeros((p1, p1))
            for dim in range(d):
                J = JT_aug_T[:, :, dim]  # shape (m, p1)
                JT_outer = J[:, None, :] * J[None, :, :]  # shape (m, m, p1)
                Lambda += np.einsum("ij,ijp,ijq->pq", K, JT_outer, JT_outer)
            Lambda /= m ** 2

        return Lambda

    def _compute_b_prior(self, JT_aug_T: np.ndarray, grad1: np.ndarray, grad2: np.ndarray) -> np.ndarray:
        """
        Computes b_prior = (1/m^2) * sum_{i,j} (J_i^T grad1_{i,j} + J_j^T grad2_{i,j})
        """
        m, p1, d = JT_aug_T.shape
        # First term: sum over i, j of JT_aug_T[i] @ grad1[i, j]
        term1 = np.einsum("ipd,ijd->p", JT_aug_T, grad1)
        # Second term: sum over i, j of JT_aug_T[j] @ grad2[i, j]
        term2 = np.einsum("jpd,ijd->p", JT_aug_T, grad2)

        return (term1 + term2) / (m ** 2)

    def _compute_b_cross(self, JT_aug_T: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Computes b_cross = (2 / m^2) * sum_{i,j} J_i^T (loss_scores[j] * K[i, j])
        """
        m, p1, d = JT_aug_T.shape
        loss_scores = self.model.posterior_score(self.samples)  # shape (m, d)
        # Multiply loss_scores[j] with K[i, j] — result shape: (m, m, d)
        weighted_scores = K[:, :, None] * loss_scores[None, :, :]  # shape (m, m, d)
        # Compute b_cross = sum_{i,j} JT_aug_T[i] @ weighted_scores[i, j]
        b_cross = np.einsum("ipd,ijd->p", JT_aug_T, weighted_scores)

        return (2 / m ** 2) * b_cross

    def _compute_C(self, trace_hess):
        m = self.samples.shape[0]
        return np.sum(trace_hess) / (m ** 2)
