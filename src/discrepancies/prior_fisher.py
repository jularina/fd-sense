from src.bayesian_model.base import BayesianModel
from src.basis_functions.basis_functions import BaseBasisFunction

from typing import Tuple
import numpy as np


class PriorFDBase:
    def __init__(self, model: "BayesianModel"):
        """
        Computes Fisher components between *prior* samples (from the reference prior)
        and a candidate prior distribution.

        Key change vs the old version:
        - Reference prior enters ONLY via its score evaluated at the samples:
              score_prior_ref(θ_i) = ∇_θ log π_ref(θ_i)
          so we do NOT require (grad_T_ref, grad_log_g_ref, eta_ref).
        """
        self.model = model
        self.samples: np.ndarray = self.model.prior_samples_init
        self.m = int(self.samples.shape[0])

        # Reference prior: only need its score at the samples
        # Shape: (m, paramdim)
        self.score_prior_ref = self.model.prior_init.grad_log_pdf(self.samples)

        # Candidate prior (exp family decomposition), evaluated at the same samples
        # grad_T: (m, paramdim, natparamdim)
        # grad_log_g: (m, paramdim)
        self.grad_T = self.model.prior_candidate.grad_sufficient_statistics(self.samples)
        self.grad_log_g = self.model.prior_candidate.grad_log_base_measure(self.samples)
        self.eta = self.model.prior_candidate.natural_parameters()

    def update_candidate(self) -> None:
        self.grad_T = self.model.prior_candidate.grad_sufficient_statistics(self.samples)
        self.grad_log_g = self.model.prior_candidate.grad_log_base_measure(self.samples)
        self.eta = self.model.prior_candidate.natural_parameters()

    # -------------------------
    # Prior-only regime
    # -------------------------

    def _v_prior_only(self) -> np.ndarray:
        """
        v_i = s_ref(θ_i) - grad_log_g(θ_i)
        where s_ref(θ) = ∇ log π_ref(θ).

        Shape: (m, paramdim)
        """
        return self.score_prior_ref - self.grad_log_g

    def _delta_score_prior_only(self, eta: np.ndarray) -> np.ndarray:
        """
        delta_i = s_ref(θ_i) - s_candidate(θ_i)
               = (s_ref(θ_i) - grad_log_g(θ_i)) - grad_T(θ_i) @ eta
               = v_i - grad_T(θ_i) @ eta

        Shape: (m, paramdim)
        """
        v = self._v_prior_only()
        gradT_eta = np.einsum("idp,p->id", self.grad_T, eta)
        return v - gradT_eta

    def estimate_fisher_prior_only(self) -> float:
        """
        (1/m) sum_i || s_ref(θ_i) - s_candidate(θ_i) ||^2
        """
        self.update_candidate()
        diff = self._delta_score_prior_only(self.eta)
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def compute_fisher_quadratic_form_prior_only(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Prior-only perturbation quadratic in eta:
            (1/m) sum_i || v_i - grad_T(θ_i) @ eta ||^2
          = eta^T A eta + b^T eta + c
        """
        self.update_candidate()
        A = self._compute_A_prior_only()
        b = self._compute_b_prior_only()
        c = self._compute_c_prior_only()
        return A, b, c

    def _compute_A_prior_only(self) -> np.ndarray:
        return np.einsum("idp,idq->pq", self.grad_T, self.grad_T) / self.m

    def _compute_b_prior_only(self) -> np.ndarray:
        v = self._v_prior_only()
        term = np.einsum("idp,id->p", self.grad_T, v)
        return (-2.0 / self.m) * term

    def _compute_c_prior_only(self) -> float:
        v = self._v_prior_only()
        return float(np.sum(v * v) / self.m)


class PriorFDNonParametric(PriorFDBase):
    def __init__(self, model: "BayesianModel"):
        """
        Fisher divergence between priors in the nonparametric representation:
            (1/l) sum_i || s_{pi_ref}(theta_i) - gradT(theta_i) @ eta ||^2

        Reference enters ONLY via self.score_prior_ref.
        """
        super().__init__(model=model)

    def _v_prior_only(self) -> np.ndarray:
        """
        Nonparametric prior-only case (Appendix):
            v_i = s_{pi_ref}(theta_i)

        Shape: (m, paramdim)  [here m corresponds to l in the write-up]
        """
        return self.score_prior_ref

    def compute_non_parametric_fisher_quadratic_form_prior_only(
        self,
        basis_func: "BaseBasisFunction",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Quadratic form for:
            (1/l) sum_i || s_{pi_ref}(theta_i) - gradT(theta_i) @ eta ||^2

        Returns:
            A_c: (K, K)
            b_c: (K,)
            c_c: float
        """
        # For the nonparametric form, grad_T comes from the basis (not prior_candidate).
        self.grad_T = self._compute_grad_basis_function_for_prior(basis_func)

        A = self._compute_A_prior_only()
        b = self._compute_b_prior_only()
        c = self._compute_c_prior_only()
        return A, b, c

    def _compute_grad_basis_function_for_prior(
        self,
        basis_func: "BaseBasisFunction",
    ) -> np.ndarray:
        """
        gradT(theta_i) = Jacobian of T(theta) = [phi_1, ..., phi_K]^T at samples.

        Expected shape: (m, paramdim, K)
        """
        return basis_func.gradient(self.samples)

