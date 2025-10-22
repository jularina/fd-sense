import numpy as np
from typing import Tuple

from src.discrepancies.fisher import FisherDivergence
from src.bayesian_model.base import BayesianModel
from src.utils.typing import ArrayLike
from src.basis_functions.basis_functions import BaseBasisFunction


class PosteriorFDBase:
    def __init__(
        self,
        samples: ArrayLike,
        model: BayesianModel,
        candidate_type: str = "prior",
    ):
        """
        Computes Fisher components between posterior samples
        and a candidate posterior distribution.
        """
        self.samples: np.ndarray = samples
        self.model: BayesianModel = model
        self.prior_scores_ref = self.model.reference_prior_score(self.samples)
        self.loss_scores_ref = self.model.reference_loss_score(self.samples)

        if candidate_type == "prior":
            self.fisher: FisherDivergence = FisherDivergence(self.prior_scores_ref, model.prior_score)
        elif candidate_type == "loss":
            self.fisher: FisherDivergence = FisherDivergence(self.loss_scores_ref, model.loss_score)

    def estimate_fisher(self) -> float:
        """
        Compute Fisher Divergence between the posterior samples
        and the candidate posterior.
        """
        return self.fisher.compute(self.samples)

    def compute_fisher_quadratic_form_for_prior(
        self,
    ) -> Tuple:
        """
        Compute components of the FD quadratic form specific to the prior term.
        """
        JT_aug_T = self._compute_augmented_jacobians_for_prior()
        Lambda_prior = self._compute_Lambda_for_prior(JT_aug_T)
        b_prior = self._compute_b_prior(JT_aug_T)

        return Lambda_prior, b_prior

    def compute_fisher_quadratic_form_for_loss(
        self,
    ) -> Tuple:
        """
        Compute components of the FD quadratic form specific to the prior term.
        """
        scores_loss = self.model.loss_score(self.samples, multiply_by_lr=False)
        Lambda_loss = self._compute_Lambda_for_loss(scores_loss)
        b_loss = self._compute_b_loss(scores_loss)

        return Lambda_loss, b_loss

    def _compute_Lambda_for_loss(self, scores: np.ndarray) -> float:
        """
        Compute Lambda term for the loss KSD quadratic form.

        Returns:
            float: Scalar Lambda value
        """
        Lambda = np.einsum('ik,jk->ij', scores, scores)
        return np.sum(Lambda) / (2*self.model.m)

    def _compute_b_loss(self, scores: np.ndarray) -> np.ndarray:
        """
        Computes b_prior = (1/m^2) * sum_{i,j} (J_i^T grad1_{i,j} + J_j^T grad2_{i,j})
        """
        term = np.einsum('ik,jk->ij', scores, self.loss_scores_ref)

        return -1*np.sum(term) / (self.model.m)

    def compute_c_loss(self) -> np.ndarray:
        """
        Computes prior constant
        """
        term = np.einsum('ik,jk->ij', self.loss_scores_ref, self.loss_scores_ref)

        return 1*np.sum(term) / (2*self.model.m)

    def compute_c_prior(self) -> np.ndarray:
        """
        Computes prior constant
        """
        term = np.einsum('ik,jk->ij', self.prior_scores_ref, self.prior_scores_ref)

        return 1*np.sum(term) / (2*self.model.m)

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


class PosteriorFDNonParametric(PosteriorFDBase):
    def __init__(
        self,
        samples: ArrayLike,
        model: BayesianModel,
        candidate_type: str = "prior",
    ):
        """
        Computes FD components between posterior samples
        and a candidate posterior distribution for nonparametric case.
        """
        super().__init__(samples=samples, model=model, candidate_type=candidate_type)

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

        phi = basis_func.evaluate(samples)  # (m, d, K)
        grad_phi = basis_func.gradient(samples)  # (m, d, K)
        grad_phi_T = np.transpose(grad_phi, (0, 2, 1))  # (m, K, d)

        return grad_phi_T