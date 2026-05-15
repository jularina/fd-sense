from src.bayesian_model.base import BayesianModel
from src.basis_functions.basis_functions import BaseBasisFunction

from typing import Tuple
import numpy as np


class PriorFDBase:
    def __init__(self, model: "BayesianModel"):
        """
        Base class for Fisher divergence between priors.

        Holds prior samples and the reference prior score. Subclasses add
        either a parametric candidate prior (PriorFDParametric) or a
        nonparametric basis-function representation (PriorFDNonParametric).
        """
        self.model = model
        self.samples: np.ndarray = self.model.prior_samples_init
        self.m = int(self.samples.shape[0])

        # Reference prior score at the samples — shape: (m, paramdim)
        self.score_prior_ref = self.model.prior_init.grad_log_pdf(self.samples)

    # -------------------------
    # Shared quadratic-form helpers (require self.grad_T and _v_prior_only)
    # -------------------------

    def _compute_A_prior_only(self) -> np.ndarray:
        return np.einsum("idp,idq->pq", self.grad_T, self.grad_T) / self.m

    def _compute_b_prior_only(self) -> np.ndarray:
        v = self._v_prior_only()
        term = np.einsum("idp,id->p", self.grad_T, v)
        return (-2.0 / self.m) * term

    def _compute_c_prior_only(self) -> float:
        v = self._v_prior_only()
        return float(np.sum(v * v) / self.m)


class PriorFDParametric(PriorFDBase):
    def __init__(self, model: "BayesianModel"):
        """
        Fisher divergence between priors in the parametric (exponential-family) representation.

        Requires model.prior_candidate to be set.
        """
        super().__init__(model=model)

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

    def _v_prior_only(self) -> np.ndarray:
        """
        v_i = s_ref(θ_i) - grad_log_g(θ_i)

        Shape: (m, paramdim)
        """
        return self.score_prior_ref - self.grad_log_g

    def _delta_score_prior_only(self, eta: np.ndarray) -> np.ndarray:
        """
        delta_i = s_ref(θ_i) - s_candidate(θ_i)
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


class PriorFDNonParametric(PriorFDBase):
    def __init__(self, model: "BayesianModel"):
        """
        Fisher divergence between priors in the nonparametric representation.
        No candidate prior required. The base measure g is the reference prior.
        """
        super().__init__(model=model)
        # ∇_θ log g(θ_i) where g = prior_init is the base measure in the nonparametric family
        self.grad_log_g = self.model.prior_init.grad_log_pdf(self.samples)

    def _v_prior_only(self) -> np.ndarray:
        return self.score_prior_ref - self.grad_log_g

    def compute_non_parametric_fisher_quadratic_form_prior_only(
        self,
        basis_func: "BaseBasisFunction",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Quadratic form using nonparametric basis functions.
        """
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
