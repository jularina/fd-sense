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

    def compute_ksd_quadratic_form_for_prior(self):
        JT_aug_T = self._compute_augmented_jacobians()  # (m, p+1, d)
        Lambda_m = self._compute_Lambda(JT_aug_T)
        b_prior = self._compute_b_prior(JT_aug_T)
        b_cross = self._compute_b_cross(JT_aug_T)
        b_m = b_prior + b_cross
        C = self._compute_C()
        return Lambda_m, b_m, b_prior, b_cross, C, JT_aug_T

    def _compute_augmented_jacobians(self):
        J_T = self.model.jacobian_sufficient_statistics(self.samples)  # (m, d, p)
        s_h = self.model.grad_log_base_measure(self.samples)  # (m, d)
        JT_aug = np.concatenate([J_T, s_h[..., None]], axis=2)  # (m, d, p+1)
        JT_aug_T = np.transpose(JT_aug, (0, 2, 1))  # (m, p+1, d)
        return JT_aug_T

    def _compute_Lambda(self, JT_aug_T: np.ndarray) -> np.ndarray:
        """
        Computes Lambda_m = (1/m^2) sum_{i,j} J_i @ J_j^T * k(theta_i, theta_j)
        """
        m, p1, d = JT_aug_T.shape
        Lambda = np.einsum('ij,ipd,jqd->pq', self.kernel.value, JT_aug_T, JT_aug_T)

        return Lambda / (m ** 2)

    def _compute_b_prior(self, JT_aug_T: np.ndarray) -> np.ndarray:
        """
        Computes b_prior = (1/m^2) * sum_{i,j} (J_i^T grad1_{i,j} + J_j^T grad2_{i,j})
        """
        m, p1, d = JT_aug_T.shape
        # First term: sum over i, j of JT_aug_T[i] @ grad1[i, j]
        term1 = np.einsum("ipd,ijd->p", JT_aug_T, self.kernel.grad_x1)
        # Second term: sum over i, j of JT_aug_T[j] @ grad2[i, j]
        term2 = np.einsum("jpd,ijd->p", JT_aug_T, self.kernel.grad_x2)

        return (term1 + term2) / (m ** 2)

    def _compute_b_cross(self, JT_aug_T: np.ndarray) -> np.ndarray:
        """
        Computes b_cross = (2 / m^2) * sum_{i,j} J_i^T (loss_scores[j] * K[i, j])
        """
        m, p1, d = JT_aug_T.shape
        loss_scores = self.model.loss_score(self.samples)  # shape (m, d)
        b_cross = np.einsum("ipd,jd,ij->p", JT_aug_T, loss_scores, self.kernel.value)

        return (2 / m ** 2) * b_cross

    def _compute_C(self):
        m = self.samples.shape[0]
        return np.sum(self.kernel.hess_xy) / (m ** 2)

    def compute_ksd_for_loss_term(self):
        m, d = self.samples.shape
        scores = self.model.loss_score(self.samples)

        K = self.kernel.value          # (m, m)
        grad1 = self.kernel.grad_x1        # (m, m, d)
        grad2 = self.kernel.grad_x2        # (m, m, d)
        hess = self.kernel.hess_xy         # (m, m) or (m, m, d, d)

        if hess.ndim == 2:
            term4 = hess  # univariate: (m, m)
        elif hess.ndim == 4:
            term4 = np.trace(hess, axis1=-2, axis2=-1)  # multivariate: (m, m)
        else:
            raise ValueError(f"Unexpected hessian shape: {hess.shape}")

        term1 = np.einsum('ik,jk,ij->ij', scores, scores, K)
        term2 = np.einsum('ik,ijk->ij', scores, grad2)
        term3 = np.einsum('jk,ijk->ij', scores, grad1)

        total = term1 + term2 + term3 + term4
        return np.sum(total) / (m * m)

    def compute_ksd_for_prior_term(self):
        m, d = self.samples.shape
        scores = self.model.prior_score(self.samples)

        K = self.kernel.value          # (m, m)
        grad1 = self.kernel.grad_x1        # (m, m, d)
        grad2 = self.kernel.grad_x2        # (m, m, d)
        hess = self.kernel.hess_xy         # (m, m) or (m, m, d, d)

        if hess.ndim == 2:
            term4 = hess  # univariate: (m, m)
        elif hess.ndim == 4:
            term4 = np.trace(hess, axis1=-2, axis2=-1)  # multivariate: (m, m)
        else:
            raise ValueError(f"Unexpected hessian shape: {hess.shape}")

        term1 = np.einsum('ik,jk,ij->ij', scores, scores, K)
        term2 = np.einsum('ik,ijk->ij', scores, grad2)
        term3 = np.einsum('jk,ijk->ij', scores, grad1)
        total = term1 + term2 + term3 + term4

        return np.sum(total) / (m * m)

    def compute_cross_term(self) -> float:
        """
        Computes the cross-term of the posterior KSD:
            (2 / m^2) * sum_{i,j} prior_score[i]^T * loss_score[j] * k(theta_i, theta_j)
        """
        m, d = self.samples.shape
        prior_scores = self.model.prior_score(self.samples)  # shape (m, d)
        loss_scores = self.model.loss_score(self.samples)    # shape (m, d)
        K = self.kernel.value                                # shape (m, m)

        term = np.einsum('ik,jk,ij->ij', prior_scores, loss_scores, K)

        # First cross term: s_pi(i)^T s_l(j) k(i, j)
        term1 = np.einsum('ik,jk,ij->ij', prior_scores, loss_scores, K)

        # Second cross term: s_pi(j)^T s_l(i) k(i, j)
        term2 = np.einsum('ik,jk,ij->ij', loss_scores, prior_scores, K)

        return (1.0 / m**2) * np.sum(term1 + term2)

    def compute_hessian_term(self) -> float:
        m, d = self.samples.shape
        hess = self.kernel.hess_xy         # (m, m) or (m, m, d, d)

        if hess.ndim == 2:
            term = hess  # univariate: (m, m)
        elif hess.ndim == 4:
            term = np.trace(hess, axis1=-2, axis2=-1)  # multivariate: (m, m)
        else:
            raise ValueError(f"Unexpected hessian shape: {hess.shape}")

        return np.sum(term) / (m * m)
