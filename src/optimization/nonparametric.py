from typing import Dict
import numpy as np
import cvxpy as cp
from omegaconf import OmegaConf

from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.discrepancies.posterior_ksd import PosteriorKSDNonParametric
from src.discrepancies.prior_ksd import PriorKSDNonParametric


class OptimizationNonparametricBase:
    def __init__(
        self,
        posterior_ksd: PosteriorKSDNonParametric,
        prior_ksd: PriorKSDNonParametric,
        config: Dict,
        radius_lower_bound: float = 1.0,
    ):
        """
        Base class to handle nonparametric quadratic form optimization.
        """
        self.posterior_ksd = posterior_ksd
        self.prior_ksd = prior_ksd

        basis_cls_name = config["basis_funcs_type"]
        basis_cls = BASIS_FUNCTIONS_REGISTRY[basis_cls_name]
        basis_kwargs = config.get("basis_funcs_kwargs", {})

        if basis_cls_name == "RBFBasisFunction":
            basis_kwargs = OmegaConf.to_container(basis_kwargs, resolve=True)
            basis_kwargs["samples"] = self.prior_ksd.samples

        self.basis_function = basis_cls(**basis_kwargs)
        self._validate_basis_function(self.posterior_ksd.samples.shape[1])

        self.Lambda_m_prior, self.b_m_prior, _, _ = self.posterior_ksd.compute_ksd_quadratic_form_for_nonparametric_prior(
            self.basis_function, scale_samples=config["scale_samples"])
        self.ksd_for_loss_init = self.posterior_ksd.compute_ksd_for_loss_term()
        self.C_posterior = self.posterior_ksd.compute_hessian_term()

        self.Lambda_prior, self.b_prior = self.prior_ksd.compute_ksd_quadratic_form_for_nonparametric_prior(
            self.basis_function,
            scale_samples=config["scale_samples"]
        )
        self.C_prior = self.prior_ksd.compute_hessian_term()

        # Scale-aware ridge (use trace/D to match the matrix scale)
        D = self.Lambda_prior.shape[0]
        avg_prior = np.trace(self.Lambda_prior) / max(D, 1)
        avg_obj = np.trace(self.Lambda_m_prior) / max(D, 1)

        eps_prior = max(1e-8, 1e-2 * avg_prior)
        eps_obj = max(1e-12, 1e-4 * avg_obj)

        self.Lambda_prior_reg = self.Lambda_prior + eps_prior * np.eye(D)
        self.Lambda_m_prior_reg = self.Lambda_m_prior + eps_obj * np.eye(D)

        self.r = self._compute_min_radius() + radius_lower_bound
        self.prior_ksd_val = self.prior_ksd.estimate_ksd()

    def _evaluate_prior_qf_ksd(self, eta_tilde: np.ndarray) -> float:
        return eta_tilde @ self.Lambda_m_prior_reg @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init

    def _validate_basis_function(self, dim: int):
        if not self.basis_function.check_C1(dim):
            raise ValueError("Basis function is not C^1 differentiable.")
        if not self.basis_function.check_L2(dim):
            raise ValueError("Basis function is not L2-integrable over ℝ^d.")

    def _compute_min_radius(self) -> float:
        """
        Compute the minimum value of the radius r satisfying:
            -1/4 * b_prior.T @ inv(Lambda_prior) @ b_prior + C_prior < r
        """
        try:
            Lambda_inv = np.linalg.inv(self.Lambda_prior_reg)
        except np.linalg.LinAlgError:
            raise ValueError("Lambda_prior is not invertible.")

        quad_term = -0.25 * self.b_prior.T @ Lambda_inv @ self.b_prior
        min_radius = float(quad_term + self.C_prior)

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
        objective = cp.Minimize(
            cp.trace(-self.Lambda_m_prior_reg @ Psi) - self.b_m_prior @ psi
        )

        # Constraint 1: Quadratic constraint
        constraint1 = (
            cp.trace(self.Lambda_prior_reg @ Psi)
            + self.b_prior @ psi
            + self.C_prior
            <= self.r
        )

        # Constraint 2: Schur complement LMI: [[Psi, psi], [psi.T, 1]] ⪰ 0
        schur_matrix = cp.bmat([
            [Psi, cp.reshape(psi, (D, 1))],
            [cp.reshape(psi, (1, D)), cp.Constant([[1]])]
        ])
        constraint2 = schur_matrix >> 0  # LMI constraint

        print("Condition number of Lambda_m_prior:", np.linalg.cond(self.Lambda_m_prior_reg))
        print("Condition number of Lambda_prior:", np.linalg.cond(self.Lambda_prior_reg))
        print("Norm of b_prior:", np.linalg.norm(self.b_prior))
        print("Min radius:", self.r)

        print("eig Λ_prior:", np.linalg.eigvalsh(self.Lambda_prior_reg)[:5], "… min:",
              np.min(np.linalg.eigvalsh(self.Lambda_prior_reg)))

        # Solve SDP
        problem = cp.Problem(objective, [constraint1, constraint2])
        problem.solve(solver=cp.MOSEK)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"SDP optimization failed: {problem.status}")

        ksd_est = self._evaluate_prior_qf_ksd(psi.value)
        gap = np.linalg.norm(Psi.value - np.outer(psi.value, psi.value), ord='fro')
        print("||Ψ - ψψ^T||_F:", gap)
        print("Eigenvalues of Psi:", np.linalg.eigvalsh(Psi.value))

        return {
            "objective_val": problem.value,
            "psi_opt": psi.value,
            "Psi_opt": Psi.value,
            "ksd_est": ksd_est,
        }

    def optimize_minimize_ksd(self):
        """
        Find the ψ that minimizes the KSD objective:
            ψ^T Λ_m ψ + b_m^T ψ + C

        Returns:
            A dict with optimal ψ, KSD value, and the Lambda matrix used.
        """
        try:
            Lambda_inv = np.linalg.inv(self.Lambda_m_prior_reg)
        except np.linalg.LinAlgError:
            raise ValueError("Lambda_m_prior is not invertible.")

        psi_opt = -0.5 * Lambda_inv @ self.b_m_prior
        ksd_est = self._evaluate_prior_qf_ksd(psi_opt)

        return {
            "psi_opt": psi_opt,
            "ksd_est": ksd_est,
        }
