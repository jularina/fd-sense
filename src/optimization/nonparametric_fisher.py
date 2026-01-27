from typing import Dict
import numpy as np
import cvxpy as cp
from omegaconf import OmegaConf
from scipy.linalg import eigh

from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY


class OptimisationNonparametricBase:
    def __init__(
        self,
        posterior_estimator,
        prior_estimator,
        config: Dict,
        radius: float = 0.0,
        add_nuggets: bool = False,
    ):
        """
        Base class to handle nonparametric quadratic form optimization.
        """
        self.posterior_estimator = posterior_estimator
        self.prior_estimator = prior_estimator

        basis_cls_name = config["basis_funcs_type"]
        basis_cls = BASIS_FUNCTIONS_REGISTRY[basis_cls_name]
        basis_kwargs = config.get("basis_funcs_kwargs", {})
        basis_kwargs = OmegaConf.to_container(basis_kwargs, resolve=True)
        basis_kwargs["prior_samples"] = self.prior_estimator.samples
        basis_kwargs["posterior_samples"] = self.posterior_estimator.samples
        self.basis_function = basis_cls(**basis_kwargs)

        self.A, self.b, self.c = self.posterior_estimator.compute_non_parametric_fisher_quadratic_form_prior_only(
            self.basis_function,
        )
        self.A_c, self.b_c, self.c_c = self.prior_estimator.compute_non_parametric_fisher_quadratic_form_prior_only(
            self.basis_function,
        )
        self.A = self._sym(self.A)
        self.A_c = self._sym(self.A_c)
        self.d = self.A_c.shape[0]

        if add_nuggets:
            avg_prior = np.trace(self.A_c) / max(self.d, 1)
            avg_obj = np.trace(self.A) / max(self.d, 1)
            eps_prior = max(1e-8, 1e-2 * avg_prior)
            eps_obj = max(1e-12, 1e-4 * avg_obj)
            self.A_c = self._sym(self.A_c + eps_prior * np.eye(self.d))
            self.A = self._sym(self.A + eps_obj * np.eye(self.d))

        self.r = self._compute_min_radius() + radius
        print(f"Radius {self.r}.")

    def _sym(self, A: np.ndarray) -> np.ndarray:
        return 0.5 * (A + A.T)

    @staticmethod
    def _solve_psd(A: np.ndarray, b: np.ndarray):
        """
        Tries to solve A x = b with a PSD/PD matrix A:
        - First attempt: Cholesky (fast, PD).
        - Fallback: lstsq (robust), equivalent to (near)pinv multiply.
        Returns x and a flag whether PD/cholesky succeeded.
        """
        try:
            L = np.linalg.cholesky(A)
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
            return x, True
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            return x, False

    @staticmethod
    def _pinv_psd(A: np.ndarray, rcond: float | None = None) -> np.ndarray:
        """
        Moore–Penrose pseudoinverse specialized for symmetric PSD/Hermitian:
        eigen-decompose, invert eigenvalues above tolerance.
        """
        w, V = np.linalg.eigh(A)

        if rcond is None:
            rcond = np.finfo(A.dtype).eps * max(A.shape) * max(w.max(), 1.0)
        w_inv = np.zeros_like(w)
        mask = w > rcond
        w_inv[mask] = 1.0 / w[mask]
        return (V * w_inv) @ V.T

    def _evaluate_qf(self, eta_tilde: np.ndarray) -> float:
        return eta_tilde @ self.A @ eta_tilde + self.b @ eta_tilde + self.c

    def _evaluate_constraint_qf(self, eta_tilde: np.ndarray) -> float:
        return eta_tilde @ self.A_c @ eta_tilde + self.b_c @ eta_tilde + self.c_c

    def _compute_min_radius(self) -> float:
        """
        min_eta η^T Λ_prior_reg η + b_prior^T η + C_prior
        = -1/4 b^T Λ^{-1} b + C   (PD case)
        = -1/4 b^T Λ^{+} b + C    (PSD/pinv case; valid when projecting onto range(Λ))
        """
        x, used_pd = self._solve_psd(self.A_c, self.b_c)
        if used_pd:
            quad_term = -0.25 * float(self.b_c.T @ x)
        else:
            Lp = self._pinv_psd(self.A_c)
            quad_term = -0.25 * float(self.b_c.T @ (Lp @ self.b_c))

        print(f"Computed min radius threshold: {quad_term}.")
        if quad_term < 0:
            quad_term = 0

        return float(quad_term)

    def optimize_through_sdp_relaxation(self):
        psi = cp.Variable(self.d)
        Psi = cp.Variable((self.d, self.d), symmetric=True)

        objective = cp.Minimize(
            cp.trace(-self.A @ Psi) - self.b @ psi
        )
        constraint1 = (
            cp.trace(self.A_c @ Psi)
            + self.b_c @ psi
            <= self.r
        )
        schur_matrix = cp.bmat([
            [Psi, cp.reshape(psi, (self.d, 1), order='C')],
            [cp.reshape(psi, (1, self.d), order='C'), cp.Constant([[1]])]
        ])
        constraint2 = schur_matrix >> 0

        # Diagnostics
        print("cond(A):", np.linalg.cond(self.A))
        print("cond(A_c):", np.linalg.cond(self.A_c))
        print("r:", self.r)

        problem = cp.Problem(objective, [constraint1, constraint2])
        problem.solve(solver=cp.MOSEK)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"SDP optimization failed: {problem.status}")

        primal_value = self._evaluate_qf(psi.value)
        constraint_value = self._evaluate_qf(psi.value)

        return {
            "psi_opt": psi.value,
            "Psi_opt": Psi.value,
            "primal_value": primal_value,
            "constraint_value":constraint_value,
            "dual_val": problem.value,
        }




