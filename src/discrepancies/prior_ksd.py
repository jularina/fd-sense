import numpy as np
from typing import Tuple

from src.kernels.base import BaseKernel
from src.discrepancies.ksd import KernelizedSteinDiscrepancy
from src.bayesian_model.base import BayesianModel
from src.utils.typing import ArrayLike
from src.basis_functions.basis_functions import BaseBasisFunction


class PriorKSDNonParametric:
    def __init__(
        self,
        samples: ArrayLike,
        model: BayesianModel,
        kernel: BaseKernel,
    ):
        """
        Computes KSD components for prior samples
        and a candidate prior distribution.
        """
        self.samples: np.ndarray = samples
        self.model: BayesianModel = model
        self.kernel: BaseKernel = kernel
        self.ksd: KernelizedSteinDiscrepancy = KernelizedSteinDiscrepancy(model.prior_score, kernel)

    def estimate_ksd(self) -> float:
        """
        Compute the KSD between the prior samples
        and the candidate prior.
        """
        return self.ksd.compute(self.samples)

    def compute_ksd_quadratic_form_for_nonparametric_prior(
        self,
        basis_func: BaseBasisFunction,
        scale_samples: bool = False,
    ) -> Tuple:
        """
        Compute components of the KSD quadratic form specific to the nonparametric prior term.
        """
        grad_phi_T = self._compute_grad_basis_function_for_prior(basis_func, scale_samples)
        Lambda_m = self._compute_Lambda_for_prior(grad_phi_T)
        b_prior = self._compute_b_prior(grad_phi_T)

        return Lambda_m, b_prior

    def _compute_grad_basis_function_for_prior(self, basis_func: BaseBasisFunction, scale_samples: bool = False) -> np.ndarray:
        """
        Compute basis functions gradient.
        """
        if scale_samples:
            scale = np.max(np.abs(self.samples), axis=0)
            samples = self.samples / (scale + 1e-8)
        else:
            samples = self.samples
        grad_phi = basis_func.gradient(samples)  # (m, d, K)
        grad_phi_T = np.transpose(grad_phi, (0, 2, 1))  # (m, K, d)

        return grad_phi_T

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
        term1 = np.einsum("ipd,ijd->p", JT_aug_T, self.kernel.grad_x1)
        term2 = np.einsum("jpd,ijd->p", JT_aug_T, self.kernel.grad_x2)
        return (term1 + term2) / (self.model.m ** 2)

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
