import numpy as np
from typing import Tuple

from src.kernels.base import BaseKernel
from src.discrepancies.ksd import KernelizedSteinDiscrepancy
from src.bayesian_model.base import BayesianModel
from src.utils.typing import ArrayLike
from src.basis_functions.basis_functions import BaseBasisFunction
from src.discrepancies.fisher import FisherDivergence


class PriorFDNonParametric:
    def __init__(
        self,
        samples: ArrayLike,
        model: BayesianModel,
        candidate_type: str = "prior",
    ):
        """
        Computes FD components for prior samples
        and a candidate prior distribution.
        """
        self.samples: np.ndarray = samples
        self.model: BayesianModel = model
        self.prior_scores_ref = self.model.reference_prior_score(self.samples)
        self.loss_scores_ref = self.model.reference_loss_score(self.samples)

        if candidate_type == "prior":
            self.fisher: FisherDivergence = FisherDivergence(self.prior_scores_ref, model.prior_score)
        elif candidate_type == "loss":
            self.fisher: FisherDivergence = FisherDivergence(self.loss_scores_ref, model.loss_score)

    def compute_quadratic_form_for_nonparametric_prior(
        self,
        basis_func: BaseBasisFunction,
        scale_samples: bool = False,
    ) -> Tuple:
        """
        Compute components of the KSD quadratic form specific to the nonparametric prior term.
        """
        grad_phi_T = self._compute_grad_basis_function_for_prior(basis_func, scale_samples)
        Lambda_prior = self._compute_Lambda_for_prior(grad_phi_T)
        b_prior = self._compute_b_prior(grad_phi_T)

        return Lambda_prior, b_prior

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
        return np.einsum('ipd,jqd->pq', JT_aug_T, JT_aug_T) / (2*m)

    def _compute_b_prior(self, JT_aug_T: np.ndarray) -> np.ndarray:
        """
        Compute b vector (linear term) for the prior KSD.

        Returns:
            np.ndarray: Shape (p+1,)
        """
        term = np.einsum("jpd,jd->p", JT_aug_T, self.prior_scores_ref)

        return (-1*term) / (self.model.m)