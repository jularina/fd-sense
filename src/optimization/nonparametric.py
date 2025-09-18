from typing import Dict
import numpy as np
import cvxpy as cp
from omegaconf import OmegaConf
from scipy.special import logsumexp

from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY
from src.discrepancies.posterior_ksd import PosteriorKSDNonParametric
from src.discrepancies.prior_ksd import PriorKSDNonParametric


class OptimizationNonparametricBase:
    def __init__(
        self,
        posterior_ksd: PosteriorKSDNonParametric,
        prior_ksd: PriorKSDNonParametric,
        config: Dict,
        radius_lower_bound: float = 0.0,
        precomputed_qfs: bool = False
    ):
        """
        Base class to handle nonparametric quadratic form optimization.
        """
        self.posterior_ksd = posterior_ksd
        self.prior_ksd = prior_ksd

        basis_cls_name = config["basis_funcs_type"]
        basis_cls = BASIS_FUNCTIONS_REGISTRY[basis_cls_name]
        basis_kwargs = config.get("basis_funcs_kwargs", {})

        if "RBFBasisFunction" in basis_cls_name or basis_cls_name == "SigmoidBasisFunction":
            basis_kwargs = OmegaConf.to_container(basis_kwargs, resolve=True)
            basis_kwargs["prior_samples"] = self.prior_ksd.samples
            basis_kwargs["posterior_samples"] = self.posterior_ksd.samples

        self.basis_function = basis_cls(**basis_kwargs)
        self._validate_basis_function(self.posterior_ksd.samples.shape[1])

        if not precomputed_qfs:
            self.Lambda_m_prior, self.b_m_prior, _, _ = self.posterior_ksd.compute_ksd_quadratic_form_for_nonparametric_prior(
                self.basis_function, scale_samples=config["scale_samples"])
            self.ksd_for_loss_init = self.posterior_ksd.compute_ksd_for_loss_term()
            self.C_posterior = self.posterior_ksd.compute_hessian_term()

            self.Lambda_prior, self.b_prior = self.prior_ksd.compute_ksd_quadratic_form_for_nonparametric_prior(
                self.basis_function,
                scale_samples=config["scale_samples"]
            )
            self.Lambda_prior = self._sym(self.Lambda_prior)
            self.Lambda_m_prior = self._sym(self.Lambda_m_prior)
            self.C_prior = self.prior_ksd.compute_hessian_term()
        else:
            self.Lambda_m_prior, self.b_m_prior, _, _ = self.posterior_ksd.compute_ksd_quadratic_form_for_nonparametric_prior(
                self.basis_function, scale_samples=config["scale_samples"])
            self.ksd_for_loss_init = self.posterior_ksd.compute_ksd_for_loss_term()
            self.C_posterior = self.posterior_ksd.compute_hessian_term()
            self.Lambda_prior, self.b_prior = self.prior_ksd.compute_ksd_quadratic_form_for_nonparametric_prior(
                self.basis_function,
                scale_samples=config["scale_samples"]
            )
            self.Lambda_prior = self._sym(self.Lambda_prior)
            self.Lambda_m_prior = self._sym(self.Lambda_m_prior)
            self.C_prior = self.prior_ksd.compute_hessian_term()

        # Scale-aware ridge (use trace/D to match the matrix scale)
        D = self.Lambda_prior.shape[0]
        avg_prior = np.trace(self.Lambda_prior) / max(D, 1)
        avg_obj = np.trace(self.Lambda_m_prior) / max(D, 1)

        eps_prior = max(1e-8, 1e-2 * avg_prior) # adaptive nuggets
        eps_obj = max(1e-12, 1e-4 * avg_obj)

        self.Lambda_prior_reg   = self._sym(self.Lambda_prior   + eps_prior * np.eye(D))
        self.Lambda_m_prior_reg = self._sym(self.Lambda_m_prior + eps_obj * np.eye(D))

        if config["radius_lower_bound"]:
            radius_lower_bound = config["radius_lower_bound"]

        self.r = self._compute_min_radius() + radius_lower_bound
        self.prior_ksd_val = self.prior_ksd.estimate_ksd()

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
            # Solve L y = b, then L^T x = y
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
            return x, True
        except np.linalg.LinAlgError:
            # Not PD; fall back to least-squares (implicitly uses SVD)
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            return x, False

    @staticmethod
    def _pinv_psd(A: np.ndarray, rcond: float | None = None) -> np.ndarray:
        """
        Moore–Penrose pseudoinverse specialized for symmetric PSD/Hermitian:
        eigen-decompose, invert eigenvalues above tolerance.
        """
        # Force symmetry just in case
        A = 0.5 * (A + A.T)
        w, V = np.linalg.eigh(A)
        if rcond is None:
            # adaptive cutoff ~ machine eps * size * max_eig
            rcond = np.finfo(A.dtype).eps * max(A.shape) * max(w.max(), 1.0)
        w_inv = np.zeros_like(w)
        mask = w > rcond
        w_inv[mask] = 1.0 / w[mask]
        return (V * w_inv) @ V.T

    def _evaluate_prior_qf_ksd(self, eta_tilde: np.ndarray) -> float:
        return eta_tilde @ self.Lambda_m_prior_reg @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init

    def _validate_basis_function(self, dim: int):
        if not self.basis_function.check_C1(dim):
            raise ValueError("Basis function is not C^1 differentiable.")
        if not self.basis_function.check_L2(dim):
            raise ValueError("Basis function is not L2-integrable over ℝ^d.")

    def _compute_min_radius(self) -> float:
        """
        min_eta η^T Λ_prior_reg η + b_prior^T η + C_prior
        = -1/4 b^T Λ^{-1} b + C   (PD case)
        = -1/4 b^T Λ^{+} b + C    (PSD/pinv case; valid when projecting onto range(Λ))
        """
        # Prefer solve (fast if PD); fallback to pinv
        x, used_pd = self._solve_psd(self.Lambda_prior_reg, self.b_prior)
        if used_pd:
            quad_term = -0.25 * float(self.b_prior.T @ x)
        else:
            # PSD case: use pinv explicitly
            Lp = self._pinv_psd(self.Lambda_prior_reg)
            quad_term = -0.25 * float(self.b_prior.T @ (Lp @ self.b_prior))

        print(f"Computed min radius: {quad_term}.")
        if quad_term < 0:
            quad_term = 0

        print(f"C_prior is {self.C_prior}.")
        print(f"Updates min radius: {quad_term}.")
        print(f"Chosen min radius: {quad_term + self.C_prior}.")
        return float(quad_term + self.C_prior)

    def optimize_through_sdp_relaxation(self):
        D = self.b_m_prior.shape[0]
        psi = cp.Variable(D)
        Psi = cp.Variable((D, D), symmetric=True)

        lam = 1e-2
        objective = cp.Minimize(
            cp.trace(-self.Lambda_m_prior_reg @ Psi) - self.b_m_prior @ psi + lam * cp.trace(Psi)
        )
        constraint1 = (
            cp.trace(self.Lambda_prior_reg @ Psi)
            + self.b_prior @ psi
            + self.C_prior
            <= self.r
        )
        schur_matrix = cp.bmat([
            [Psi, cp.reshape(psi, (D, 1), order='C')],
            [cp.reshape(psi, (1, D), order='C'), cp.Constant([[1]])]
        ])
        constraint2 = schur_matrix >> 0

        # Diagnostics
        print("cond(Λ_m):", np.linalg.cond(self.Lambda_m_prior_reg))
        print("cond(Λ_p):", np.linalg.cond(self.Lambda_prior_reg))
        print("||b_prior||:", np.linalg.norm(self.b_prior))
        print("r:", self.r)

        problem = cp.Problem(objective, [constraint1, constraint2])
        problem.solve(solver=cp.MOSEK)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"SDP optimization failed: {problem.status}")

        ksd_est = self._evaluate_prior_qf_ksd(psi.value)
        gap = np.linalg.norm(Psi.value - np.outer(psi.value, psi.value), ord='fro')
        print("||Ψ - ψψ^T||_F:", gap)
        print("eig(Psi) min/max:", np.min(np.linalg.eigvalsh(Psi.value)), np.max(np.linalg.eigvalsh(Psi.value)))

        return {
            "objective_val": problem.value,
            "psi_opt": psi.value,
            "Psi_opt": Psi.value,
            "ksd_est": ksd_est,
        }