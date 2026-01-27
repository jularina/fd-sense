from src.bayesian_model.base import BayesianModel
from src.basis_functions.basis_functions import BaseBasisFunction

from typing import Tuple
import numpy as np

class PriorFDBase:
    def __init__(self, model: "BayesianModel"):
        """
        Computes Fisher components between prior samples
        and candidate prior distribution
        """
        self.model = model
        self.samples: np.ndarray = self.model.prior_samples_init
        self.m = self.samples.shape[0]

        # Prior-related quantities evaluated at posterior samples
        # Shapes assumed:
        #   grad_T*:       (m, paramdim, natparamdim)
        #   grad_log_g*:   (m, paramdim)
        self.grad_T_ref = self.model.prior_init.grad_sufficient_statistics(self.samples)
        self.grad_log_g_ref = self.model.prior_init.grad_log_base_measure(self.samples)
        self.eta_ref = self.model.prior_init.natural_parameters()

        self.grad_T = self.model.prior_candidate.grad_sufficient_statistics(self.samples)
        self.grad_log_g = self.model.prior_candidate.grad_log_base_measure(self.samples)
        self.eta = self.model.prior_candidate.natural_parameters()

    def update_candidate(self):
        self.grad_T = self.model.prior_candidate.grad_sufficient_statistics(self.samples)
        self.grad_log_g = self.model.prior_candidate.grad_log_base_measure(self.samples)
        self.eta = self.model.prior_candidate.natural_parameters()

    def estimate_fisher_prior_only(self) -> float:
        """
        Estimates (1/m) sum_i || s_ref(θ_i) - s_candidate(θ_i) ||^2
        in the prior-only perturbation regime (beta fixed to beta_ref).
        Here, loss term cancels because beta is the same in ref and candidate.
        """
        self.update_candidate()
        diff = self._delta_score_prior_only(self.eta)  # (m, paramdim)
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def compute_fisher_quadratic_form_prior_only(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Prior-only perturbation quadratic form:
            (1/m) sum_i || v_i - grad_T(θ_i) @ eta ||^2
          = eta^T A eta + b^T eta + c

        Returns:
            A: (natparamdim, natparamdim)
            b: (natparamdim,)
            c: float
        """
        self.update_candidate()
        A = self._compute_A_prior_only()
        b = self._compute_b_prior_only()
        c = self._compute_c_prior_only()
        return A, b, c

    def _v_prior_only(self) -> np.ndarray:
        """
        v_i = grad_T_ref(θ_i) @ eta_ref + grad_log_g_ref(θ_i) - grad_log_g(θ_i)
        Shape: (m, paramdim)
        """
        gradTref_eta = np.einsum("idp,p->id", self.grad_T_ref, self.eta_ref)
        return gradTref_eta + self.grad_log_g_ref - self.grad_log_g

    def _delta_score_prior_only(self, eta: np.ndarray) -> np.ndarray:
        """
        delta_i = s_ref(θ_i) - s_candidate(θ_i) for prior-only case (beta fixed)
               = v_i - grad_T(θ_i) @ eta
        Shape: (m, paramdim)
        """
        v = self._v_prior_only()
        gradT_eta = np.einsum("idp,p->id", self.grad_T, eta)
        return v - gradT_eta

    def _compute_A_prior_only(self) -> np.ndarray:
        """
        A = (1/m) sum_i grad_T(θ_i)^T grad_T(θ_i)
        grad_T: (m, paramdim, natparamdim) -> A: (natparamdim, natparamdim)
        """
        return np.einsum("idp,idq->pq", self.grad_T, self.grad_T) / self.m

    def _compute_b_prior_only(self) -> np.ndarray:
        """
        b = -(2/m) sum_i grad_T(θ_i)^T v_i
        """
        v = self._v_prior_only()  # (m, paramdim)
        term = np.einsum("idp,id->p", self.grad_T, v)  # (natparamdim,)
        return (-2.0 / self.m) * term

    def _compute_c_prior_only(self) -> float:
        """
        c = (1/m) sum_i ||v_i||^2
        """
        v = self._v_prior_only()
        return float(np.sum(v * v) / self.m)


class PriorFDNonParametric(PriorFDBase):
    def __init__(
        self,
        model: BayesianModel,
    ):
        """
        Computes FD components between prior samples
        and a candidate prior distribution for nonparametric case.
        """
        super().__init__(model=model)

    def compute_non_parametric_fisher_quadratic_form_prior_only(
        self,
        basis_func: BaseBasisFunction,
    ) -> Tuple:
        """
        Compute components of the KSD quadratic form specific to the nonparametric prior term.
        """
        self.grad_T = self._compute_grad_basis_function_for_prior(basis_func)
        A = self._compute_A_prior_only()
        b = self._compute_b_prior_only()
        c = self._compute_c_prior_only()

        return A, b, c

    def _compute_grad_basis_function_for_prior(self, basis_func: BaseBasisFunction) -> np.ndarray:
        """
        Compute basis functions gradient.
        """
        grad_phi = basis_func.gradient(self.samples)

        return grad_phi

    def _v_prior_only(self) -> np.ndarray:
        """
        v_i = grad_T_ref(θ_i) @ eta_ref + grad_log_g_ref(θ_i) - grad_log_g(θ_i)
        Shape: (m, paramdim)
        """
        gradTref_eta = np.einsum("idp,p->id", self.grad_T_ref, self.eta_ref)
        return gradTref_eta + self.grad_log_g_ref