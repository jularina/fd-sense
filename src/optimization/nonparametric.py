from typing import Dict
import numpy as np
import cvxpy as cp

from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.discrepancies.posterior_ksd import PosteriorKSDNonParametric


class OptimizationNonparametricBase:
    def __init__(
        self,
        posterior_ksd: PosteriorKSDNonParametric,
        config: Dict,
    ):
        """
        Base class to handle nonparametric quadratic form optimization.
        """
        self.posterior_ksd = posterior_ksd

        basis_cls_name = config["basis_funcs_type"]
        basis_cls = BASIS_FUNCTIONS_REGISTRY[basis_cls_name]
        basis_kwargs = config.get("basis_funcs_kwargs", {})
        self.basis_function = basis_cls(**basis_kwargs)
        self._validate_basis_function(self.posterior_ksd.samples.shape[1])

        self.Lambda_m_prior, self.b_m_prior, self.b_prior, self.b_cross_prior = self.posterior_ksd.compute_ksd_quadratic_form_for_nonparametric_prior(self.basis_function)
        self.ksd_for_loss_init = self.posterior_ksd.compute_ksd_for_loss_term()
        self.C = self.posterior_ksd.compute_hessian_term()

        self.r = self._compute_min_radius() + 1.0

    def _evaluate_prior_qf_ksd(self, eta_tilde: np.ndarray) -> float:
        return eta_tilde @ self.Lambda_m_prior @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init

    def _validate_basis_function(self, dim: int):
        if not self.basis_function.check_C1(dim):
            raise ValueError("Basis function is not C^1 differentiable.")
        if not self.basis_function.check_L2(dim):
            raise ValueError("Basis function is not L2-integrable over ℝ^d.")

    def _compute_min_radius(self) -> float:
        """
        Compute the minimum value of the radius r satisfying:
            -1/4 * b_prior.T @ inv(Lambda_m_prior) @ b_prior + C_prior < r
        """
        try:
            Lambda_inv = np.linalg.inv(self.Lambda_m_prior)
        except np.linalg.LinAlgError:
            raise ValueError("Lambda_m_prior is not invertible.")

        quad_term = -0.25 * self.b_prior.T @ Lambda_inv @ self.b_prior
        min_radius = float(quad_term + self.C)

        return min_radius

    def optimize_through_sdp_relaxation(self):
        """
        Solves the SDP relaxation of the non-convex quadratic problem using a Schur complement LMI.
        Returns the optimal value and solution (ψ, Ψ) from the relaxed problem.
        """
        D = self.b_m_prior.shape[0]

        # Define variables
        psi = cp.Variable(D)
        Psi = cp.Variable((D, D), symmetric=True)

        # Objective:
        #   tr(-Lambda_m Psi) - b_m^T psi - C_m
        objective = cp.Minimize(
            cp.trace(-self.Lambda_m_prior @ Psi) - self.b_m_prior @ psi - self.ksd_for_loss_init
        )

        # Constraint 1: Quadratic constraint
        constraint1 = (
            cp.trace(self.Lambda_m_prior @ Psi)
            + self.b_prior @ psi
            + self.C
            <= self.r
        )

        # Constraint 2: Schur complement LMI: [[Psi, psi], [psi.T, 1]] ⪰ 0
        schur_matrix = cp.bmat([
            [Psi, cp.reshape(psi, (D, 1))],
            [cp.reshape(psi, (1, D)), cp.Constant([[1]])]
        ])
        constraint2 = schur_matrix >> 0  # LMI constraint

        # Solve SDP
        problem = cp.Problem(objective, [constraint1, constraint2])
        problem.solve(solver=cp.SCS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"SDP optimization failed: {problem.status}")

        ksd_est = self._evaluate_prior_qf_ksd(psi.value)

        return {
            "objective_val": problem.value,
            "psi_opt": psi.value,
            "Psi_opt": Psi.value,
        }