from src.bayesian_model.base import BayesianModel
from src.basis_functions.basis_functions import BaseBasisFunction

from typing import Tuple, List, Sequence, Dict
import numpy as np
from scipy.stats import norm


class PosteriorFDBase:
    def __init__(self, model: "BayesianModel"):
        self.model = model
        self.samples: np.ndarray = self.model.posterior_samples_init
        self.m = self.samples.shape[0]

        # Reference prior: ONLY need its score at the samples
        # Shape: (m, paramdim)
        self.score_prior_ref = self.model.prior_init.grad_log_pdf(self.samples)

        # Candidate prior (exp family decomposition)
        # grad_T: (m, paramdim, natparamdim)
        # grad_log_g: (m, paramdim)
        self.grad_T = self.model.prior_candidate.grad_sufficient_statistics(self.samples)
        self.grad_log_g = self.model.prior_candidate.grad_log_base_measure(self.samples)
        self.eta = self.model.prior_candidate.natural_parameters()

        # Loss gradient sum_j ∇_θ l(θ_i, x_j)
        self.g = self.model.loss_score(self.samples, multiply_by_lr=False)  # (m, paramdim)
        self.beta_ref = self.model.loss_lr_init
        self.beta = self.model.loss_lr

    def update_candidate(self):
        self.grad_T = self.model.prior_candidate.grad_sufficient_statistics(self.samples)
        self.grad_log_g = self.model.prior_candidate.grad_log_base_measure(self.samples)
        self.eta = self.model.prior_candidate.natural_parameters()
        self.beta = self.model.loss_lr

    # -------------------------
    # Special-case: prior-only
    # -------------------------

    def _v_prior_only(self) -> np.ndarray:
        """
        v_i = s_{pi_ref}(θ_i) - grad_log_g(θ_i)
        Shape: (m, paramdim)
        """
        return self.score_prior_ref - self.grad_log_g

    def _delta_score_prior_only(self, eta: np.ndarray) -> np.ndarray:
        """
        delta_i = s_ref(θ_i) - s_candidate(θ_i)
               = (s_{pi_ref}(θ_i) - grad_log_g(θ_i)) - grad_T(θ_i) @ eta
        Shape: (m, paramdim)
        """
        v = self._v_prior_only()
        gradT_eta = np.einsum("idp,p->id", self.grad_T, eta)
        return v - gradT_eta

    def estimate_fisher_prior_only(self) -> float:
        self.update_candidate()
        diff = self._delta_score_prior_only(self.eta)
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def estimate_fisher_for_gaussians(self) -> float:
        """
        Exact closed-form FD for Gaussian prior perturbations w.r.t. a Gaussian posterior.

        Requires both model.prior_init and model.prior_candidate to be Gaussian
        (univariate) or MultivariateGaussian, and the model to have exact posterior
        parameters (mu_n, sigma_n2 for univariate; mu_n, Sigma_n for multivariate).

        Univariate derivation
        ---------------------
        Score difference:  s_ref(θ) - s_cand(θ) = a·θ + b
          a = 1/σ_cand² - 1/σ_ref²
          b = μ_ref/σ_ref² - μ_cand/σ_cand²
        FD = E[( aθ + b )²] = a²(σ_n² + μ_n²) + 2ab·μ_n + b²

        Multivariate derivation
        -----------------------
        Score difference:  δ(θ) = A·θ + b_vec
          A     = Σ_cand⁻¹ - Σ_ref⁻¹
          b_vec = Σ_ref⁻¹ μ_ref - Σ_cand⁻¹ μ_cand
        FD = tr(AᵀA (Σ_n + μ_n μ_nᵀ)) + 2 b_vec·A μ_n + ‖b_vec‖²
        """
        from src.distributions.gaussian import Gaussian, MultivariateGaussian

        pi_ref = self.model.prior_init
        pi_cand = self.model.prior_candidate

        if isinstance(pi_ref, Gaussian) and isinstance(pi_cand, Gaussian):
            a = 1.0 / pi_cand.var - 1.0 / pi_ref.var
            b = pi_ref.mu / pi_ref.var - pi_cand.mu / pi_cand.var
            mu_n = float(self.model.mu_n)
            sigma_n2 = float(self.model.sigma_n2)
            return float(a ** 2 * (sigma_n2 + mu_n ** 2) + 2.0 * a * b * mu_n + b ** 2)

        if isinstance(pi_ref, MultivariateGaussian) and isinstance(pi_cand, MultivariateGaussian):
            A = pi_cand.cov_inv - pi_ref.cov_inv
            b_vec = pi_ref.cov_inv @ pi_ref.mu - pi_cand.cov_inv @ pi_cand.mu
            mu_n = np.asarray(self.model.mu_n)
            Sigma_n = np.asarray(self.model.Sigma_n)
            AtA = A.T @ A
            second_moment = Sigma_n + np.outer(mu_n, mu_n)
            return float(np.trace(AtA @ second_moment) + 2.0 * b_vec @ A @ mu_n + b_vec @ b_vec)

        raise TypeError(
            "estimate_fisher_for_gaussians requires both prior_init and prior_candidate "
            "to be Gaussian or MultivariateGaussian instances."
        )

    def fd_prior_only_given_eta(self, eta: np.ndarray) -> float:
        """
        Black-box objective: compute \hat{rho}^FD_m for prior-only perturbations,
        evaluated at the provided natural parameter vector eta.
        """
        self.update_candidate()  # refresh cached grad_T, grad_log_g, etc.
        eta = np.asarray(eta, dtype=float).reshape(-1)

        # delta_i = (s_{pi_ref}(θ_i) - grad_log_g(θ_i)) - grad_T(θ_i) @ eta
        v = self._v_prior_only()  # (m, paramdim)
        gradT_eta = np.einsum("idp,p->id", self.grad_T, eta)  # (m, paramdim)
        diff = v - gradT_eta
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def compute_fisher_quadratic_form_prior_only(
        self,
    ) -> Tuple:
        """
        Prior-only perturbation quadratic in eta:
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

    # -------------------------------
    # Special-case: learning-rate-only
    # -------------------------------

    def estimate_fisher_lr_only(self) -> float:
        """
        Learning-rate-only perturbation with eta fixed to eta_ref (and prior fixed to ref).
        Then:
            s_ref(θ_i) - s_beta(θ_i) = (beta - beta_ref) * g_i
        so FD = (1/m) sum_i ||(beta - beta_ref) g_i||^2
        """
        self.update_candidate()
        diff = (self.beta - self.beta_ref) * self.g
        return float(np.mean(np.sum(diff * diff, axis=1)))

    def compute_fisher_quadratic_form_lr_only(
        self,
    ) -> Tuple[float, float, float]:
        """
        Learning-rate-only perturbation quadratic in beta:
            FD(beta) = A beta^2 + b beta + c
        with:
            A = (1/m) sum_i ||g_i||^2
            b = -2 beta_ref A
            c = beta_ref^2 A

        Returns:
            A: float
            b: float
            c: float
        """
        self.update_candidate()
        A = float(np.mean(np.sum(self.g * self.g, axis=1)))
        b = float(-2.0 * self.beta_ref * A)
        c = float((self.beta_ref ** 2) * A)
        return A, b, c

    def compute_prior_only_qf_per_component(
            self,
            component_names: List[str],
            theta_blocks: Sequence[Sequence[int]],
            eta_blocks: Sequence[Sequence[int]],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Build per-component quadratic forms:
            Q_j(eta_j) = eta_j^T A_j eta_j + b_j^T eta_j + c_j
        where each component j uses only theta coordinates in theta_blocks[j]
        and only natural-parameter coordinates in eta_blocks[j].

        This matches the true blockwise squared-score objective for factorised priors.

        Returns:
          dict name -> (A_j, b_j, c_j)
        """
        assert len(component_names) == len(theta_blocks) == len(eta_blocks)

        self.update_candidate()
        v = self._v_prior_only()  # (m, paramdim)
        G = self.grad_T  # (m, paramdim, natparamdim)
        m = self.m

        out = {}
        for name, th_block, et_block in zip(component_names, theta_blocks, eta_blocks):
            th_block = list(th_block)
            et_block = list(et_block)
            v_b = v[:, th_block]  # (m, tb)
            G_b = G[:, th_block, :][:, :, et_block]  # (m, tb, eb)
            A_j = np.einsum("itp,itq->pq", G_b, G_b) / m  # (eb, eb)
            b_j = (-2.0 / m) * np.einsum("itp,it->p", G_b, v_b)  # (eb,)
            c_j = float(np.mean(np.sum(v_b * v_b, axis=1)))
            out[name] = (A_j, b_j, c_j)

        return out

    def fd_gaussian_copula_given_lambda(
            self,
            lam: float,
            idx_g0: int = 0,
            idx_nu: int = 2,
            eps: float = 1e-10,
    ) -> float:
        """
        Empirical Fisher divergence for Gaussian copula perturbation
        when posterior samples are already rescaled to (0,1).

        Computes
            E_{theta ~ Pi_ref} || ∇_theta log c_lam(u_G0, u_nu) ||^2.

        Parameters
        ----------
        lam : float
            Gaussian copula correlation parameter. Must satisfy |lam| < 1.
        idx_g0 : int
            Column index of the rescaled G0 coordinate in self.samples.
        idx_nu : int
            Column index of the rescaled nu coordinate in self.samples.
        eps : float
            Boundary clipping level for numerical stability.

        Returns
        -------
        float
            Monte-Carlo estimate of the Fisher divergence.
        """
        lam = float(lam)
        if abs(lam) >= 1.0:
            return np.inf

        u_g0 = np.asarray(self.samples[:, idx_g0], dtype=float)
        u_nu = np.asarray(self.samples[:, idx_nu], dtype=float)

        # numerical stabilisation near boundaries
        u_g0 = np.clip(u_g0, eps, 1.0 - eps)
        u_nu = np.clip(u_nu, eps, 1.0 - eps)

        z_g0 = norm.ppf(u_g0)
        z_nu = norm.ppf(u_nu)

        phi_z_g0 = norm.pdf(z_g0)
        phi_z_nu = norm.pdf(z_nu)

        denom = 1.0 - lam ** 2

        # derivatives w.r.t. unit-interval coordinates
        dlogc_dg0 = lam * (z_nu - lam * z_g0) / (denom * phi_z_g0)
        dlogc_dnu = lam * (z_g0 - lam * z_nu) / (denom * phi_z_nu)

        sq_norm = dlogc_dg0 ** 2 + dlogc_dnu ** 2

        return float(np.mean(sq_norm))

    def diagnose_gaussian_copula_l2(
            self,
            lam: float,
            *,
            idx_g0: int = 0,
            idx_nu: int = 2,
            eps: float = 0.0,
    ) -> Dict[str, float]:
        """
        Empirical diagnostics for square-integrability of the Gaussian copula score
        under posterior samples already rescaled to the unit interval.

        The function estimates
            E[ || ∇_θ log c_lam(u_G0, u_nu) ||^2 ]
        by Monte Carlo and reports whether the score is finite on the sampled points.

        Parameters
        ----------
        lam : float
            Gaussian copula correlation parameter. Must satisfy |lam| < 1.
        idx_g0 : int, default=0
            Column index of the rescaled G0 coordinate in self.samples.
        idx_nu : int, default=2
            Column index of the rescaled nu coordinate in self.samples.
        eps : float, default=0.0
            Optional clipping level. If eps > 0, then u is clipped to
            [eps, 1 - eps] before applying the Gaussian quantile transform.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
              - lambda
              - n_samples
              - finite_fraction
              - mean_sq_score
              - max_sq_score
              - q95_sq_score
              - q99_sq_score
              - q999_sq_score
              - u_g0_min, u_g0_max
              - u_nu_min, u_nu_max
              - u_g0_min_boundary_dist
              - u_nu_min_boundary_dist
        """
        lam = float(lam)
        if abs(lam) >= 1.0:
            raise ValueError("lam must lie in (-1, 1).")

        u_g0 = np.asarray(self.samples[:, idx_g0], dtype=float).reshape(-1)
        u_nu = np.asarray(self.samples[:, idx_nu], dtype=float).reshape(-1)

        if eps > 0.0:
            u_g0 = np.clip(u_g0, eps, 1.0 - eps)
            u_nu = np.clip(u_nu, eps, 1.0 - eps)

        z_g0 = norm.ppf(u_g0)
        z_nu = norm.ppf(u_nu)

        phi_z_g0 = norm.pdf(z_g0)
        phi_z_nu = norm.pdf(z_nu)

        denom = 1.0 - lam ** 2

        # Samples are already in (0,1), so there are no extra affine scaling factors.
        dlogc_dg0 = lam * (z_nu - lam * z_g0) / (denom * phi_z_g0)
        dlogc_dnu = lam * (z_g0 - lam * z_nu) / (denom * phi_z_nu)

        sq_norm = dlogc_dg0 ** 2 + dlogc_dnu ** 2
        finite_mask = np.isfinite(sq_norm)

        finite_vals = sq_norm[finite_mask]

        return {
            "lambda": lam,
            "n_samples": int(sq_norm.shape[0]),
            "finite_fraction": float(np.mean(finite_mask)),
            "mean_sq_score": float(np.mean(finite_vals)) if finite_vals.size > 0 else np.nan,
            "max_sq_score": float(np.max(finite_vals)) if finite_vals.size > 0 else np.nan,
            "q95_sq_score": float(np.quantile(finite_vals, 0.95)) if finite_vals.size > 0 else np.nan,
            "q99_sq_score": float(np.quantile(finite_vals, 0.99)) if finite_vals.size > 0 else np.nan,
            "q999_sq_score": float(np.quantile(finite_vals, 0.999)) if finite_vals.size > 0 else np.nan,
            "u_g0_min": float(np.min(u_g0)),
            "u_g0_max": float(np.max(u_g0)),
            "u_nu_min": float(np.min(u_nu)),
            "u_nu_max": float(np.max(u_nu)),
            "u_g0_min_boundary_dist": float(np.min(np.minimum(u_g0, 1.0 - u_g0))),
            "u_nu_min_boundary_dist": float(np.min(np.minimum(u_nu, 1.0 - u_nu))),
        }


class PosteriorFDNonParametric(PosteriorFDBase):
    def __init__(self, model: BayesianModel):
        super().__init__(model=model)

    def _v_prior_only(self) -> np.ndarray:
        """
        Nonparametric prior-only case (Appendix):
            v_i = s_{pi_ref}(theta_i)
        Shape: (m, paramdim)
        """
        return self.score_prior_ref

    def compute_non_parametric_fisher_quadratic_form_prior_only(
        self,
        basis_func: BaseBasisFunction,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Quadratic form for:
            (1/m) sum_i || s_{pi_ref}(theta_i) - gradT(theta_i) @ eta ||^2

        Here grad_T is the Jacobian of T(theta) = [phi_1(theta), ..., phi_K(theta)].
        So grad_T has shape (m, paramdim, K).
        """
        # IMPORTANT: do NOT call update_candidate() here, because for the nonparametric
        # representation grad_T comes from the basis, not from prior_candidate.
        self.grad_T = self._compute_grad_basis_function_for_prior(basis_func)

        A = self._compute_A_prior_only()   # uses self.grad_T
        b = self._compute_b_prior_only()   # uses v = score_prior_ref (via override)
        c = self._compute_c_prior_only()   # uses v = score_prior_ref (via override)

        return A, b, c

    def _compute_grad_basis_function_for_prior(
        self,
        basis_func: BaseBasisFunction
    ) -> np.ndarray:
        """
        Returns grad_phi with shape (m, paramdim, K).
        """
        grad_phi = basis_func.gradient(self.samples)
        return grad_phi
