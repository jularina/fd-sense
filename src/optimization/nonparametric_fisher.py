from typing import Dict
import numpy as np
import cvxpy as cp
from omegaconf import OmegaConf

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
        psd, rank = self._check_pd_psd_and_rank(self.A)
        print(f"A is {psd} with rank {rank}.")
        psd, rank = self._check_pd_psd_and_rank(self.A_c)
        print(f"A_c is {psd} with rank {rank}.")

        self.d = self.A_c.shape[0]

        if add_nuggets:
            avg_prior = np.trace(self.A_c) / max(self.d, 1)
            avg_obj = np.trace(self.A) / max(self.d, 1)
            eps_prior = max(1e-8, 1e-2 * avg_prior)
            eps_obj = max(1e-12, 1e-4 * avg_obj)
            self.A_c = self._sym(self.A_c + eps_prior * np.eye(self.d))
            self.A = self._sym(self.A + eps_obj * np.eye(self.d))
            psd, rank = self._check_pd_psd_and_rank(self.A)
            print(f"After nuggets A is {psd} with rank {rank}.")
            psd, rank = self._check_pd_psd_and_rank(self.A_c)
            print(f"After nuggets A_c is {psd} with rank {rank}.")

        self.r = self._compute_min_radius() + radius
        print(f"Radius {self.r}.")

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

    def _check_pd_psd_and_rank(
            self,
            A: np.ndarray,
            tol: float = 1e-10,
    ):
        """
        Check whether a symmetric matrix is positive definite (PD),
        positive semidefinite (PSD), or not PSD, and return its numerical rank.
        """
        eigvals = np.linalg.eigvalsh(A)
        rank = int(np.sum(eigvals > tol))

        if np.all(eigvals > tol):
            status = "PD"
        elif np.all(eigvals >= -tol):
            status = "PSD"
        else:
            status = "NOT_PSD"

        return status, rank

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
        quad_term = -0.25 * float(self.b_c.T @ self._pinv_psd(self.A_c) @ self.b_c)
        min_val = float(self.c_c + quad_term)
        print(f"Computed min radius threshold: {min_val}.")
        return max(0.0, min_val)

    def optimize_through_sdp_relaxation(self):
        psi = cp.Variable(self.d)
        Psi = cp.Variable((self.d, self.d), symmetric=True)

        objective = cp.Maximize(cp.trace(self.A @ Psi) + self.b @ psi + self.c)
        constraint1 = (
            cp.trace(self.A_c @ Psi) + self.b_c @ psi + self.c_c <= self.r
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
        constraint_value = self._evaluate_constraint_qf(psi.value)

        return {
            "eta_star": psi.value,
            "Psi_opt": Psi.value,
            "primal_value": primal_value,
            "constraint_value": constraint_value,
            "dual_value": problem.value,
        }

    # ----------------------------
    # Helpers for 1D dual in lambda
    # ----------------------------

    def _M_s(self, lam: float):
        M = lam * self.A_c - self.A
        s = lam * self.b_c - self.b
        return M, s

    def _eta_from_lambda(self, lam: float, tol: float) -> np.ndarray:
        M, s = self._M_s(lam)
        M_pinv = np.linalg.pinv(M, rcond=tol)
        return -0.5 * (M_pinv @ s)

    def _constraint(self, eta: np.ndarray) -> float:
        return float(eta.T @ self.A_c @ eta + self.b_c.T @ eta + self.c_c)

    def _objective(self, eta: np.ndarray) -> float:
        return float(eta.T @ self.A @ eta + self.b.T @ eta + self.c)

    def _max_eig_sym(self, X: np.ndarray) -> float:
        Xs = 0.5 * (X + X.T)
        return float(np.linalg.eigvalsh(Xs).max())

    def _range_ok(self, M: np.ndarray, s: np.ndarray, tol: float) -> bool:
        M_pinv = np.linalg.pinv(M, rcond=tol)
        resid = s - M @ (M_pinv @ s)
        return float(np.linalg.norm(resid)) <= 1e3 * tol * (1.0 + float(np.linalg.norm(s)))

    def _dual_1d_value_strict(self, lam: float, radius: float, tol: float) -> float:
        """
        Dual objective:
            d(lam) = c - lam(c_c - r) - 1/4 s^T M^† s
        Strictly return +inf if domain conditions fail.
        """
        if lam < 0:
            return float("inf")

        r = float(radius)
        M, s = self._M_s(lam)

        # domain: M(lam) ⪯ 0 and s in Range(M)
        if self._max_eig_sym(M) > 1e3 * tol:
            return float("inf")
        if not self._range_ok(M, s, tol):
            return float("inf")

        M_pinv = np.linalg.pinv(M, rcond=tol)
        quad = float(s.T @ (M_pinv @ s))  # should be <= 0 if M ⪯ 0
        return float(self.c - lam * (self.c_c - r) - 0.25 * quad)

    def _bracket_feasible_lambda(self, radius: float, tol: float, lam_max: float, grid: int):
        """
        Find an interval [L,U] that contains feasible lambdas (domain ok).
        """
        xs = np.linspace(0.0, lam_max, grid)
        feas = []
        for lam in xs:
            val = self._dual_1d_value_strict(lam, radius, tol)
            feas.append(np.isfinite(val))
        feas = np.array(feas, dtype=bool)
        if not feas.any():
            return None

        idx = np.where(feas)[0]
        # take the first contiguous feasible run
        start = idx[0]
        end = start
        for j in idx[1:]:
            if j == end + 1:
                end = j
            else:
                break
        L = float(xs[start])
        U = float(xs[end])

        # widen a little if single point
        if start == end:
            if start > 0:
                L = float(xs[start - 1])
            if start < len(xs) - 1:
                U = float(xs[start + 1])
        return (max(0.0, L), max(0.0, U))

    def _refine_lambda_by_active_constraint(
        self,
        lam_init: float,
        radius: float,
        tol: float,
        lam_lo: float,
        lam_hi: float,
        max_iter: int = 80,
    ) -> float:
        """
        If lambda>0, KKT suggests g(eta(lambda)) = r.
        Enforce by bisection on phi(lam) = g(eta(lam)) - r over a feasible bracket.
        Assumes phi is continuous on the feasible interval.
        """
        r = float(radius)

        def phi(lam: float) -> float:
            eta = self._eta_from_lambda(lam, tol)
            return self._constraint(eta) - r

        # Ensure endpoints feasible for eta map (domain should already be ok via bracket)
        plo = phi(lam_lo)
        phi_ = phi(lam_hi)

        # If no sign change, just return lam_init (cannot bisect safely).
        if not np.isfinite(plo) or not np.isfinite(phi_) or plo * phi_ > 0:
            return float(lam_init)

        a, b = lam_lo, lam_hi
        fa, fb = plo, phi_

        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = phi(m)
            if not np.isfinite(fm):
                break
            if abs(fm) <= 1e2 * tol:
                return float(m)
            # bisection
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm

        return float(0.5 * (a + b))

    def is_singular_symmetric(self, M, tol=1e-10):
        eigvals = np.linalg.eigvalsh(0.5 * (M + M.T))
        return np.min(np.abs(eigvals)) <= tol

    # ---------- main method ----------
    def _sym(self, A: np.ndarray) -> np.ndarray:
        return 0.5 * (A + A.T)

    def optimize_through_dual_1d_lambda(
            self,
            tol: float = 1e-10,
            lam_init: float | None = None,
            lam_cap: float = 1e12,
            max_expand: int = 80,
            max_iter: int = 200,
    ):
        """
        Solve   inf_{λ>=0} d(λ)  with domain:
          M(λ)=λ A_c - A ⪰ 0  and  s(λ)=λ b_c - b ∈ Range(M(λ)).

        Uses:
          - eigendecomp-based feasibility check + pseudoinverse
          - bracketing by expansion to get a 3-point "bowl"
          - golden-section search on the bracket (convex, 1D)
        """

        A = self._sym(self.A)
        Ac = self._sym(self.A_c)
        b = np.asarray(self.b).reshape(-1)
        bc = np.asarray(self.b_c).reshape(-1)
        c0 = float(self.c)
        cc = float(self.c_c)
        r = float(self.r)
        n = A.shape[0]

        # ---------- helpers ----------
        def eval_dual_and_eta(lam: float):
            """
            Returns (feasible: bool, dval: float, eta: np.ndarray | None).
            If infeasible -> (False, +inf, None).
            """
            if lam < 0.0:
                return False, float("inf"), None

            M = self._sym(lam * Ac - A)
            s = lam * bc - b

            # Eigendecomp for symmetric M
            w, V = np.linalg.eigh(M)
            wmax = float(w.max()) if w.size else 0.0

            # PSD test tolerance (scale-aware)
            eps = 1e3 * tol * max(1.0, abs(wmax))
            if float(w.min()) < -eps:
                return False, float("inf"), None

            # Range condition: s ⟂ Null(M)
            null_mask = w <= eps
            if np.any(null_mask):
                U0 = V[:, null_mask]
                proj = U0.T @ s
                if float(np.linalg.norm(proj)) > 1e3 * tol * (1.0 + float(np.linalg.norm(s))):
                    return False, float("inf"), None

            # pseudoinverse via eigvals
            w_inv = np.zeros_like(w)
            pos_mask = w > eps
            w_inv[pos_mask] = 1.0 / w[pos_mask]
            M_pinv = (V * w_inv) @ V.T

            quad = float(s @ (M_pinv @ s))  # >= 0 if M ⪰ 0
            dval = float(c0 - lam * (cc - r) + 0.25 * quad)

            eta = -0.5 * (M_pinv @ s)
            return True, dval, eta

        def find_lambda_min_feasible():
            """
            Find the smallest feasible lambda by:
              1) exponential search to find some feasible point
              2) bisection back to the boundary
            """
            if lam_init is None:
                lam_hi = 1.0
            else:
                lam_hi = max(0.0, float(lam_init))

            feas, _, _ = eval_dual_and_eta(lam_hi)
            expand = 0
            while not feas and lam_hi < lam_cap and expand < max_expand:
                lam_hi = 2.0 * lam_hi if lam_hi > 0 else 1.0
                feas, _, _ = eval_dual_and_eta(lam_hi)
                expand += 1

            if not feas:
                return None  # no feasible point found

            lam_lo = 0.0
            # bisection to boundary (lam_hi feasible, lam_lo may be infeasible)
            for _ in range(max_iter):
                mid = 0.5 * (lam_lo + lam_hi)
                feas_mid, _, _ = eval_dual_and_eta(mid)
                if feas_mid:
                    lam_hi = mid
                else:
                    lam_lo = mid
                if abs(lam_hi - lam_lo) <= 1e-14 * (1.0 + abs(lam_hi) + abs(lam_lo)):
                    break
            return float(lam_hi)

        def bracket_minimizer(lam0: float):
            """
            Produce a bracket [a,b,c] with a < b < c and f(b) <= f(a), f(b) <= f(c).
            Convex function => minimizer lies in [a,c].
            """
            # Start at feasible boundary
            a = lam0
            fa_ok, fa, _ = eval_dual_and_eta(a)
            if not fa_ok:
                return None

            # Step right
            step = max(1e-6, 0.05 * (1.0 + a))
            b = a + step
            fb_ok, fb, _ = eval_dual_and_eta(b)

            # Ensure b feasible; if not, push until feasible
            tries = 0
            while (not fb_ok) and b < lam_cap and tries < max_expand:
                step *= 2.0
                b = a + step
                fb_ok, fb, _ = eval_dual_and_eta(b)
                tries += 1
            if not fb_ok:
                return None

            # If already increasing right away, convex min is at boundary a
            if fb >= fa:
                return (a, a, b, fa, fa, fb)  # degenerate bowl: min at a

            # Expand to the right until we see an upturn: f(c) >= f(b)
            c = b + step
            fc_ok, fc, _ = eval_dual_and_eta(c)
            tries = 0
            while (not fc_ok or fc < fb) and c < lam_cap and tries < max_expand:
                # If infeasible, jump further right (domain typically becomes feasible for larger λ)
                step *= 2.0
                a, fa = b, fb
                b, fb = c, fc if fc_ok else fb
                c = b + step
                fc_ok, fc, _ = eval_dual_and_eta(c)
                tries += 1

            if not fc_ok:
                # We couldn't find a right point feasible+upturn, but we may still proceed with what we have
                # In practice, boundedness assumption should prevent this.
                return None

            return (a, b, c, fa, fb, fc)

        def golden_section(a: float, b: float, c: float):
            """
            Minimize on [a,c] assuming unimodal/convex.
            If b==a, returns a.
            """
            if b == a:
                return a

            left, right = a, c
            gr = 0.5 * (np.sqrt(5.0) - 1.0)  # golden ratio conjugate

            x1 = right - gr * (right - left)
            x2 = left + gr * (right - left)

            f1_ok, f1, _ = eval_dual_and_eta(x1)
            f2_ok, f2, _ = eval_dual_and_eta(x2)

            # If feasibility fails inside, nudge inward; but with convex feasible set this should be rare
            for _ in range(max_iter):
                if (right - left) <= 1e-12 * (1.0 + abs(left) + abs(right)):
                    break

                # If either point infeasible, shrink interval toward feasible side
                if not f1_ok:
                    left = x1
                    x1 = right - gr * (right - left)
                    f1_ok, f1, _ = eval_dual_and_eta(x1)
                    continue
                if not f2_ok:
                    right = x2
                    x2 = left + gr * (right - left)
                    f2_ok, f2, _ = eval_dual_and_eta(x2)
                    continue

                if f1 > f2:
                    left = x1
                    x1 = x2
                    f1 = f2
                    x2 = left + gr * (right - left)
                    f2_ok, f2, _ = eval_dual_and_eta(x2)
                else:
                    right = x2
                    x2 = x1
                    f2 = f1
                    x1 = right - gr * (right - left)
                    f1_ok, f1, _ = eval_dual_and_eta(x1)

            return 0.5 * (left + right)

        # ---------- main ----------
        lam0 = find_lambda_min_feasible()
        if lam0 is None:
            return {
                "lambda_star": np.nan,
                "dual_value": np.nan,
                "eta_star": np.full(n, np.nan),
                "status": "no_feasible_lambda_found",
            }

        bowl = bracket_minimizer(lam0)
        if bowl is None:
            feas0, d0, eta0 = eval_dual_and_eta(lam0)
            return {
                "lambda_star": float(lam0),
                "dual_value": float(d0),
                "eta_star": np.asarray(eta0) if eta0 is not None else np.full(n, np.nan),
                "status": "failed_to_bracket_minimizer_return_boundary",
            }

        a, b, c, fa, fb, fc = bowl

        # If degenerate bowl indicates boundary minimum
        if b == a:
            lam_star = a
        else:
            lam_star = golden_section(a, b, c)

        feas, dstar, eta_star = eval_dual_and_eta(lam_star)

        def dual_value(lam: float) -> float:
            M = self._sym(lam * Ac - A)
            s = lam * bc - b
            M_pinv = np.linalg.pinv(M, rcond=tol)
            quad = float(s @ (M_pinv @ s))
            return float(c0 - lam * (cc - r) + 0.25 * quad)

        dual_val = float(dual_value(lam_star))

        return {
            "lambda_star": float(lam_star),
            "dual_value": float(dstar),
            "eta_star": np.asarray(eta_star) if eta_star is not None else np.full(n, np.nan),
            "domain_feasible": bool(feas),
            "lambda_min_feasible": float(lam0),
            "bracket": (float(a), float(b), float(c)),
            "status": "optimal_dual_1d_golden",
        }

    def optimize_dual_sdp_lambda_t(
        self,
        solver=None,
        solver_opts=None,
        tol: float = 1e-8,
    ):
        """
        Solve the single-constraint QCQP via the SDP dual in (lambda, t):

            min_{lambda >= 0, t}  t + c - lambda (c_c - radius)
            s.t.  [[lambda A_c - A, 1/2 (b - lambda b_c)],
                  [1/2 (b - lambda b_c)^T, t]]  ⪰ 0

        Then recover a primal maximiser via the KKT stationarity condition:
            2(A - lambda A_c) eta + (b - lambda b_c) = 0
            => eta = -1/2 * pinv(A - lambda A_c) * (b - lambda b_c)

        Returns DualQCQPSolution with primal/dual values and feasibility diagnostics.
        """
        try:
            import cvxpy as cp
        except Exception as e:
            raise ImportError(
                "cvxpy is required for solve_dual_sdp_lambda_t(). "
                "Install it (and a solver like SCS/ECOS/MOSEK)."
            ) from e

        n = self.A.shape[0]
        if self.A.shape != (n, n) or self.A_c.shape != (n, n):
            raise ValueError("A and A_c must be square matrices of the same shape.")
        if self.b.shape != (n,) or self.b_c.shape != (n,):
            raise ValueError("b and b_c must be vectors of length natparamdim.")

        r = self.r

        lam = cp.Variable(nonneg=True)   # lambda >= 0
        t = cp.Variable()                # free scalar

        # Build LMI:
        top_left = lam * self.A_c - self.A
        top_right = 0.5 * (self.b - lam * self.b_c)
        LMI = cp.bmat([
            [top_left,              cp.reshape(top_right, (n, 1))],
            [cp.reshape(top_right, (1, n)), cp.reshape(t, (1, 1))]
        ])

        objective = cp.Minimize(t + self.c - lam * (self.c_c - r))
        constraints = [LMI >> 0]

        prob = cp.Problem(objective, constraints)

        solve_kwargs = {}
        if solver is not None:
            solve_kwargs["solver"] = solver
        if solver_opts:
            solve_kwargs.update(solver_opts)

        try:
            prob.solve(**solve_kwargs)
        except Exception:
            # Fallback: SCS is a reasonable default for SDPs
            prob.solve(solver=cp.SCS)

        status = str(prob.status)

        # If the solve failed, return a structured "failed" solution
        if prob.value is None or lam.value is None or t.value is None:
            eta_nan = np.full(n, np.nan, dtype=float)
            return {}

        lambda_star = float(lam.value)
        dual_value = float(prob.value)

        # Domain objects for diagnostics (your new notation)
        M = lambda_star * self.A_c - self.A
        s = lambda_star * self.b_c - self.b
        r = self.is_singular_symmetric(M)
        eta_star = -0.5 * np.linalg.inv(M) @ s

        Q = lambda_star * self.A_c - self.A  # should be PSD
        s_dual = lambda_star * self.b_c - self.b  # should lie in Range(Q)
        Q_pinv = np.linalg.pinv(Q, rcond=tol)
        resid = s_dual - Q @ (Q_pinv @ s_dual)

        min_eig_Q = float(np.linalg.eigvalsh(0.5 * (Q + Q.T)).min())
        range_ok = float(np.linalg.norm(resid)) <= 1e3 * tol * (1.0 + float(np.linalg.norm(s_dual)))

        # Don't pretend we recovered primal maximiser:
        # eta_star = np.full(n, np.nan, dtype=float)
        primal_value = eta_star @ self.A @ eta_star + eta_star @ self.b + self.c

        return {
            "dual_value": dual_value,
            "eta_star": eta_star,
            "primal_value": primal_value,
            "lambda_star": lambda_star,
        }
