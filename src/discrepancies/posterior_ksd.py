import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from src.kernels.base import BaseKernel
from src.discrepancies.ksd import KernelizedSteinDiscrepancy
from src.bayesian_model.base import BayesianModel
from src.utils.typing import ArrayLike
from src.basis_functions.basis_functions import BaseBasisFunction


class PosteriorKSDBase:
    def __init__(
        self,
        samples: ArrayLike,
        model: BayesianModel,
        kernel: BaseKernel,
    ):
        """
        Computes KSD components between posterior samples
        and a candidate posterior distribution.
        """
        self.samples: np.ndarray = samples
        self.model: BayesianModel = model
        self.kernel: BaseKernel = kernel
        self.ksd: KernelizedSteinDiscrepancy = KernelizedSteinDiscrepancy(model.posterior_score, kernel)

    def estimate_ksd(self) -> float:
        """
        Compute the KSD between the posterior samples
        and the candidate posterior.
        """
        return self.ksd.compute(self.samples)

    def compute_ksd_quadratic_form_for_loss(
        self,
    ) -> Tuple:
        """
        Compute components of the KSD quadratic form specific to the likelihood term.
        """
        scores_loss = self.model.loss_score(self.samples, multiply_by_lr=False)
        scores_prior = self.model.prior_score(self.samples)
        Lambda_m = self._compute_Lambda_for_loss(scores_loss)
        b_loss = self._compute_b_loss(scores_loss)
        b_cross = self._compute_b_cross_for_loss(scores_loss, scores_prior)
        b_m = b_loss + b_cross
        return Lambda_m, b_m, b_loss, b_cross

    def _compute_Lambda_for_loss(self, scores: np.ndarray) -> float:
        """
        Compute Lambda term for the loss KSD quadratic form.

        Returns:
            float: Scalar Lambda value
        """
        Lambda = np.einsum('ik,jk,ij->ij', scores, scores, self.kernel.value)
        return np.sum(Lambda) / (self.model.m ** 2)

    def _compute_b_loss(self, scores: np.ndarray) -> np.ndarray:
        """
        Computes b_prior = (1/m^2) * sum_{i,j} (J_i^T grad1_{i,j} + J_j^T grad2_{i,j})
        """
        term1 = np.einsum('ik,ijk->ij', scores, self.kernel.grad_x2)
        term2 = np.einsum('jk,ijk->ij', scores, self.kernel.grad_x1)

        return np.sum(term1 + term2) / (self.model.m ** 2)

    def _compute_b_cross_for_loss(self, scores_loss: np.ndarray, scores_prior: np.ndarray) -> float:
        """
        Compute cross-term b value for the loss KSD.

        Returns:
            float: Scalar value
        """
        term = np.einsum('ik,jk,ij->ij', scores_loss, scores_prior, self.kernel.value)
        return (2 / self.model.m ** 2) * np.sum(term)

    def compute_ksd_for_loss_term(self) -> float:
        """
        Direct computation of the KSD value for the loss term.

        Returns:
            float: KSD value for likelihood component
        """
        m, d = self.samples.shape
        scores = self.model.loss_score(self.samples)

        K = self.kernel.value
        grad1 = self.kernel.grad_x1
        grad2 = self.kernel.grad_x2
        hess = self.kernel.hess_xy

        if hess.ndim == 2:
            term4 = hess
        elif hess.ndim == 4:
            term4 = np.trace(hess, axis1=-2, axis2=-1)
        else:
            raise ValueError(f"Unexpected hessian shape: {hess.shape}")

        term1 = np.einsum('ik,jk,ij->ij', scores, scores, K)
        term2 = np.einsum('ik,ijk->ij', scores, grad2)
        term3 = np.einsum('jk,ijk->ij', scores, grad1)

        return np.sum(term1 + term2 + term3 + term4) / (m ** 2)

    def compute_ksd_for_prior_term(self) -> float:
        """
        Direct computation of the KSD value for the prior term.

        Returns:
            float: KSD value for prior component
        """
        m, d = self.samples.shape
        scores = self.model.prior_score(self.samples)

        K = self.kernel.value
        grad1 = self.kernel.grad_x1
        grad2 = self.kernel.grad_x2
        hess = self.kernel.hess_xy

        if hess.ndim == 2:
            term4 = hess
        elif hess.ndim == 4:
            term4 = np.trace(hess, axis1=-2, axis2=-1)
        else:
            raise ValueError(f"Unexpected hessian shape: {hess.shape}")

        term1 = np.einsum('ik,jk,ij->ij', scores, scores, K)
        term2 = np.einsum('ik,ijk->ij', scores, grad2)
        term3 = np.einsum('jk,ijk->ij', scores, grad1)

        return np.sum(term1 + term2 + term3 + term4) / (m ** 2)

    def compute_cross_term(self) -> float:
        """
        Compute the cross term of the posterior KSD: 2 * s_pi^T s_l * k

        Returns:
            float: Scalar value
        """
        m, d = self.samples.shape
        prior_scores = self.model.prior_score(self.samples)
        loss_scores = self.model.loss_score(self.samples)
        K = self.kernel.value

        term1 = np.einsum('ik,jk,ij->ij', prior_scores, loss_scores, K)
        term2 = np.einsum('ik,jk,ij->ij', loss_scores, prior_scores, K)

        return (1.0 / m**2) * np.sum(term1 + term2)

    def compute_hessian_term(self) -> float:
        """
        Compute the Hessian trace term of the KSD (scalar).

        Returns:
            float: Scalar value
        """
        m, d = self.samples.shape
        hess = self.kernel.hess_xy

        if hess.ndim == 2:
            return np.sum(hess) / (m ** 2)
        elif hess.ndim == 4:
            return np.sum(np.trace(hess, axis1=-2, axis2=-1)) / (m ** 2)
        else:
            raise ValueError(f"Unexpected hessian shape: {hess.shape}")

    def _compute_Lambda_for_prior(self, JT_aug_T: np.ndarray) -> np.ndarray:
        """
        Compute the Lambda matrix for the prior KSD quadratic form.

        Returns:
            np.ndarray: Shape (p+1, p+1)
        """
        m, p, d = JT_aug_T.shape
        return np.einsum('ij,ipd,jqd->pq', self.kernel.value, JT_aug_T, JT_aug_T) / (m ** 2)

    def _compute_b_prior(self, JT_aug_T: np.ndarray) -> np.ndarray:
        """
        Compute b vector (linear term) for the prior KSD.

        Returns:
            np.ndarray: Shape (p+1,)
        """
        term1 = np.einsum("jpd,ijd->p", JT_aug_T, self.kernel.grad_x1)
        term2 = np.einsum("ipd,ijd->p", JT_aug_T, self.kernel.grad_x2)

        return (term1 + term2) / (self.model.m ** 2)

    def _compute_b_cross_for_prior(self, JT_aug_T: np.ndarray) -> np.ndarray:
        """
        Compute cross-term b vector for the prior KSD.

        Returns:
            np.ndarray: Shape (p+1,)
        """
        loss_scores = self.model.loss_score(self.samples)
        b_cross = np.einsum("ipd,jd,ij->p", JT_aug_T, loss_scores, self.kernel.value)
        return (2 / self.model.m ** 2) * b_cross


class PosteriorKSDParametric(PosteriorKSDBase):
    def __init__(
        self,
        samples: ArrayLike,
        model: BayesianModel,
        kernel: BaseKernel,
    ):
        """
        Computes KSD components between posterior samples
        and a candidate posterior distribution for parametric case.
        """
        super().__init__(samples=samples, model=model, kernel=kernel)

    def compute_ksd_quadratic_form_for_prior(
        self,
    ) -> Tuple:
        """
        Compute components of the KSD quadratic form specific to the prior term.
        """
        JT_aug_T = self._compute_augmented_jacobians_for_prior()
        Lambda_m = self._compute_Lambda_for_prior(JT_aug_T)
        b_prior = self._compute_b_prior(JT_aug_T)
        b_cross = self._compute_b_cross_for_prior(JT_aug_T)
        b_m = b_prior + b_cross
        return Lambda_m, b_m, b_prior, b_cross

    def _compute_augmented_jacobians_for_prior(self) -> np.ndarray:
        """
        Compute augmented Jacobians including sufficient statistics and base measure gradient.

        Returns:
            np.ndarray: Augmented Jacobians of shape (m, p+1, d)
        """
        J_T = self.model.jacobian_sufficient_statistics(self.samples)  # (m, d, p)
        s_h = self.model.grad_log_base_measure(self.samples)  # (m, d)
        JT_aug = np.concatenate([J_T, s_h[..., None]], axis=2)  # (m, d, p+1)
        JT_aug_T = np.transpose(JT_aug, (0, 2, 1))  # (m, p+1, d)
        return JT_aug_T


class PosteriorKSDNonParametric(PosteriorKSDBase):
    def __init__(
        self,
        samples: ArrayLike,
        model: BayesianModel,
        kernel: BaseKernel,
    ):
        """
        Computes KSD components between posterior samples
        and a candidate posterior distribution for nonparametric case.
        """
        super().__init__(samples=samples, model=model, kernel=kernel)

    def compute_ksd_quadratic_form_for_nonparametric_prior(
        self,
        basis_func: BaseBasisFunction,
        scale_samples: bool = False,
    ) -> Tuple:
        """
        Compute components of the KSD quadratic form specific to the nonparametric prior term.
        """
        grad_phi_T = self._compute_grad_basis_function_for_prior(basis_func, scale_samples)
        J = grad_phi_T
        Lambda_m = self._compute_Lambda_for_prior(grad_phi_T)
        b_prior = self._compute_b_prior(grad_phi_T)
        b_cross = self._compute_b_cross_for_prior(grad_phi_T)
        b_m = b_prior + b_cross

        return Lambda_m, b_m, b_prior, b_cross

    def _compute_grad_basis_function_for_prior(self, basis_func: BaseBasisFunction, scale_samples: bool = False) -> np.ndarray:
        """
        Compute basis functions gradient.
        """
        if scale_samples:
            mean = np.mean(self.samples, axis=0)
            std = np.std(self.samples, axis=0)
            samples = (self.samples - mean) / (std + 1e-8)
        else:
            samples = self.samples

        phi = basis_func.evaluate(samples)  # (m, d, K)
        grad_phi = basis_func.gradient(samples)  # (m, d, K)

        # plot_basis_stuff(self.samples, phi, grad_phi)

        grad_phi_T = np.transpose(grad_phi, (0, 2, 1))  # (m, K, d)

        return grad_phi_T


def plot_basis_stuff(samples, phi, grad_phi):
    samples = np.asarray(samples)
    phi = np.asarray(phi)
    grad_phi = np.asarray(grad_phi)

    if samples.ndim != 2 or samples.shape[1] != 1:
        raise ValueError(f"samples should be (m, 1); got {samples.shape}")
    if phi.ndim != 3 or phi.shape[1] != 1:
        raise ValueError(f"phi should be (m, 1, K); got {phi.shape}")
    if grad_phi.ndim != 3 or grad_phi.shape[1] != 1:
        raise ValueError(f"grad_phi should be (m, 1, K); got {grad_phi.shape}")

    m = samples.shape[0]
    K = phi.shape[2]

    samples_1d = samples.squeeze(1)    # (m,)
    phi_K = phi.squeeze(1)             # (m, K)
    grad_phi_K = grad_phi.squeeze(1)   # (m, K)

    # Sort by sample value for smooth lines
    sort_idx = np.argsort(samples_1d)
    xs = samples_1d[sort_idx]
    phi_sorted = phi_K[sort_idx]
    grad_phi_sorted = grad_phi_K[sort_idx]

    # 1) Samples scatter (just points along x-axis)
    plt.figure()
    plt.scatter(samples_1d, np.zeros_like(samples_1d), marker='o', s=20)
    plt.title("Samples")
    plt.xlabel("Sample value")
    plt.yticks([])
    plt.grid(True)

    # 2) Phi with sample positions marked
    plt.figure()
    for k in range(K):
        plt.plot(xs, phi_sorted[:, k], label=f"Basis {k}")
    plt.scatter(samples_1d, np.zeros_like(samples_1d), color='black', marker='x', s=30, label="Samples")
    plt.title("Basis Functions (phi)")
    plt.xlabel("Sample value")
    plt.ylabel("phi value")
    plt.grid(True)
    plt.legend()

    # 3) Grad Phi with sample positions marked
    plt.figure()
    for k in range(K):
        plt.plot(xs, grad_phi_sorted[:, k], label=f"Basis {k}")
    plt.scatter(samples_1d, np.zeros_like(samples_1d), color='black', marker='x', s=30, label="Samples")
    plt.title("Gradients of Basis Functions (grad_phi)")
    plt.xlabel("Sample value")
    plt.ylabel("Gradient value")
    plt.grid(True)
    plt.legend()

    plt.show()
