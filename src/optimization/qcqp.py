from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from typing import Dict, Any
import numpy as np


@dataclass
class QCQPSolution:
    eta_star: np.ndarray          # (p,)
    x_star: np.ndarray            # (p,) = eta_star - eta_ref
    lambda_star: float            # leading generalized eigenvalue
    achieved_constraint: float    # x_star^T A_c x_star
    achieved_objective: float     # x_star^T A x_star


@dataclass
class DualQCQPSolution:
    eta_star: np.ndarray
    lambda_star: float
    primal_value: float
    dual_value: float
    constraint_value: float
    feasible: bool
    dual_feasible: bool
    status: str


def _pick_cvxpy_solver() -> str:
    # Prefer commercial/fast solvers if available, fallback to SCS
    try:
        import cvxpy as cp
        installed = cp.installed_solvers()
        for s in ["MOSEK", "GUROBI", "CPLEX", "ECOS", "SCS"]:
            if s in installed:
                return s
        return "SCS"
    except Exception:
        return "SCS"


class ParametricQCQPBase:
    """
    Closed-form solver for the prior-only, same-distribution-family special case
    """

    def __init__(self, posterior_fd, prior_fd, symmetrize: bool = True, alpha: float = 1e-6):
        A_c, b_c, c_c = prior_fd.compute_fisher_quadratic_form_prior_only()
        A, b, c = posterior_fd.compute_fisher_quadratic_form_prior_only()

        n = A_c.shape[0]
        eps = alpha * (np.trace(A_c) / n)
        A_c = A_c + eps * np.eye(n)

        self.A_c = np.array(A_c, dtype=float)
        self.b_c = np.array(b_c, dtype=float).reshape(-1)
        self.c_c = float(c_c)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float).reshape(-1)
        self.c = float(c)
        self.eta_ref = prior_fd.model.prior_init.natural_parameters()

        if symmetrize:
            self.A_c = 0.5 * (self.A_c + self.A_c.T)
            self.A = 0.5 * (self.A + self.A.T)

    def solve_generalized_eigenvalue(
        self,
        r: float,
        tol: float = 1e-12,
        check_kernel_condition: bool = False,
    ) -> QCQPSolution:
        """
        Solves sup_{x^T A_c x <= r} x^T A x via generalized eigenvalue.

        Args:
            r: radius (must be >= 0)
            tol: eigenvalue threshold for rank detection in A_c
            check_kernel_condition: optionally checks ker(A_c) ⊆ ker(A) numerically

        Returns:
            QCQPSolution with eta_star, x_star, etc.
        """
        if r <= 0:
            raise ValueError(f"r must be nonnegative, got r={r}")

        # Handle trivial radius
        p = self.A_c.shape[0]

        # Eigendecomposition of A_c
        evals, evecs = np.linalg.eigh(self.A_c)  # symmetric
        pos = evals > tol
        rank = int(np.sum(pos))

        if rank == 0:
            raise ValueError(
                "A_c is (numerically) zero / rank-0; the feasible set is unbounded "
                "and the eigenvalue closed-form does not apply."
            )

        # Restrict to range R(A_c): U columns are eigenvectors with positive evals
        U = evecs[:, pos]                 # (p, rank)
        Lam = evals[pos]                  # (rank,)
        Lambda = np.diag(Lam)             # (rank, rank)

        # Optional numeric kernel condition check: ker(A_c) ⊆ ker(A)
        if check_kernel_condition and rank < p:
            U0 = evecs[:, ~pos]           # basis for ker(A_c)
            # We want A U0 ~ 0
            resid = np.linalg.norm(self.A @ U0, ord="fro")
            if resid > 1e-8 * max(1.0, np.linalg.norm(self.A, ord="fro")):
                raise ValueError(
                    "Kernel condition ker(A_c) ⊆ ker(A) appears violated numerically. "
                    f"Residual ||A U0||_F = {resid:.3e}."
                )

        # Reduced matrix: M = U^T A U
        M = U.T @ self.A @ U              # (rank, rank)
        M = 0.5 * (M + M.T)               # symmetrize for numerical stability

        # Solve generalized eigenproblem: M y = λ Λ y with Λ ≻ 0 diagonal
        # Convert to standard eigenproblem using whitening:
        #   B = Λ^{-1/2} M Λ^{-1/2}, then B v = λ v, and y = Λ^{-1/2} v.
        inv_sqrt_Lam = np.diag(1.0 / np.sqrt(Lam))  # (rank, rank)
        B = inv_sqrt_Lam @ M @ inv_sqrt_Lam
        B = 0.5 * (B + B.T)

        w, V = np.linalg.eigh(B)
        idx = int(np.argmax(w))
        lambda_star = float(w[idx])
        v_star = V[:, idx]  # (rank,)

        y_star = inv_sqrt_Lam @ v_star    # (rank,)

        # Normalise so that y^T Λ y = 1 (equivalently z^T A_c z = 1 for z=Uy)
        norm = float(y_star.T @ Lambda @ y_star)
        if norm <= 0:
            raise ValueError("Numerical issue: computed non-positive norm for constraint normalization.")
        y_star = y_star / np.sqrt(norm)

        # Recover z* in R(A_c), then x* = sqrt(r) z*
        z_star = U @ y_star               # (p,)
        x_star = np.sqrt(r) * z_star
        eta_star = self.eta_ref + x_star

        achieved_constraint = float(x_star.T @ self.A_c @ x_star)
        achieved_objective = float(x_star.T @ self.A @ x_star)

        return QCQPSolution(
            eta_star=eta_star,
            x_star=x_star,
            lambda_star=lambda_star,
            achieved_constraint=achieved_constraint,
            achieved_objective=achieved_objective,
        )

    def solve_dual_sdp_lambda_t(
        self,
        radius: float,
        solver: Optional[str] = None,
        solver_opts: Optional[Dict[str, Any]] = None,
        tol: float = 1e-8,
    ) -> DualQCQPSolution:
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

        r = float(radius)

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
            return DualQCQPSolution(
                eta_star=eta_nan,
                lambda_star=float("nan"),
                primal_value=float("nan"),
                dual_value=float("nan"),
                constraint_value=float("nan"),
                feasible=False,
                dual_feasible=False,
                status=status,
            )

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
        dual_feasible = (lambda_star >= -1e3 * tol) and (min_eig_Q >= -1e3 * tol) and range_ok

        # Don't pretend we recovered primal maximiser:
        # eta_star = np.full(n, np.nan, dtype=float)
        primal_value = eta_star @ self.A @ eta_star + eta_star @ self.b + self.c
        constraint_value = float("nan")
        feasible = True

        return DualQCQPSolution(
            eta_star=eta_star,
            lambda_star=lambda_star,
            primal_value=primal_value,
            dual_value=dual_value,
            constraint_value=constraint_value,
            feasible=feasible,
            dual_feasible=dual_feasible,
            status=status,
        )

    # ----------------------------
    # Helpers for 1D dual in lambda
    # ----------------------------

    def _M_s(self, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        M = self.A - lam * self.A_c
        s = self.b - lam * self.b_c
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

    def _bracket_feasible_lambda(self, radius: float, tol: float, lam_max: float, grid: int) -> Optional[Tuple[float, float]]:
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

    def solve_dual_1d_lambda(
            self,
            radius: float,
            tol: float = 1e-8,
            lam_max: Optional[float] = None,
            grid_size: int = 500,
            max_iter: int = 80,
    ) -> DualQCQPSolution:
        n = self.A.shape[0]
        r = float(radius)

        A = self._sym(self.A)
        Ac = self._sym(self.A_c)
        b = np.asarray(self.b).reshape(-1)
        bc = np.asarray(self.b_c).reshape(-1)
        c0 = float(self.c)
        cc = float(self.c_c)

        def constraint_val(eta: np.ndarray) -> float:
            eta = eta.reshape(-1)
            return float(eta @ (Ac @ eta) + bc @ eta + cc)

        def objective_val(eta: np.ndarray) -> float:
            eta = eta.reshape(-1)
            return float(eta @ (A @ eta) + b @ eta + c0)

        def dual_domain_ok(lam: float) -> bool:
            if lam < 0.0:
                return False
            M = self._sym(lam * Ac - A)
            s = lam * bc - b

            # PSD check
            w = np.linalg.eigvalsh(M)
            if float(w.min()) < -1e3 * tol:
                return False

            # Range check: s in Range(M)  <=>  projection onto Null(M) is ~0
            # Use eigendecomp for stability
            null_mask = w <= (1e3 * tol) * max(1.0, float(w.max()))
            if np.any(null_mask):
                U0 = np.linalg.eigh(M)[1][:, null_mask]  # columns spanning approx nullspace
                proj = U0.T @ s
                if float(np.linalg.norm(proj)) > 1e3 * tol * (1.0 + float(np.linalg.norm(s))):
                    return False
            return True

        def dual_value(lam: float) -> float:
            if not dual_domain_ok(lam):
                return float("inf")
            M = self._sym(lam * Ac - A)
            s = lam * bc - b
            M_pinv = np.linalg.pinv(M, rcond=tol)
            quad = float(s @ (M_pinv @ s))
            return float(c0 - lam * (cc - r) + 0.25 * quad)

        # ---- Find smallest feasible lambda (left boundary of dual domain) ----
        # First bracket where domain becomes feasible.
        def find_first_feasible_interval(lam_hi: float, grid: int) -> Optional[tuple[float, float]]:
            xs = np.linspace(0.0, lam_hi, grid)
            feas = np.array([dual_domain_ok(float(x)) for x in xs], dtype=bool)
            if not feas.any():
                return None
            k = int(np.where(feas)[0][0])
            lo = float(xs[max(k - 1, 0)])
            hi = float(xs[k])
            return (lo, hi)

        if lam_max is None:
            lam_hi = 1.0
            bracket = None
            for _ in range(16):
                bracket = find_first_feasible_interval(lam_hi, grid=max(200, grid_size // 2))
                if bracket is not None:
                    break
                lam_hi *= 2.0
            if bracket is None:
                eta_nan = np.full(n, np.nan, dtype=float)
                return DualQCQPSolution(
                    eta_star=eta_nan,
                    lambda_star=float("nan"),
                    primal_value=float("nan"),
                    dual_value=float("nan"),
                    constraint_value=float("nan"),
                    feasible=False,
                    dual_feasible=False,
                    status="no_dual_feasible_lambda_found",
                )
        else:
            bracket = find_first_feasible_interval(float(lam_max), grid=grid_size)
            if bracket is None:
                eta_nan = np.full(n, np.nan, dtype=float)
                return DualQCQPSolution(
                    eta_star=eta_nan,
                    lambda_star=float("nan"),
                    primal_value=float("nan"),
                    dual_value=float("nan"),
                    constraint_value=float("nan"),
                    feasible=False,
                    dual_feasible=False,
                    status="no_dual_feasible_lambda_in_[0,lam_max]",
                )

        lam_lo, lam_hi = bracket

        # Refine minimal feasible lambda by bisection on domain feasibility
        for _ in range(max_iter):
            mid = 0.5 * (lam_lo + lam_hi)
            if dual_domain_ok(mid):
                lam_hi = mid
            else:
                lam_lo = mid
            if abs(lam_hi - lam_lo) <= 1e-12 * (1.0 + abs(lam_hi) + abs(lam_lo)):
                break

        lambda_star = float(lam_hi)  # minimal feasible
        dual_val = float(dual_value(lambda_star))

        # ---- Primal recovery ----
        M = self._sym(lambda_star * Ac - A)
        s = lambda_star * bc - b

        # Try PD solve first
        w, V = np.linalg.eigh(M)
        wmax = float(w.max())
        eps_null = (1e3 * tol) * max(1.0, wmax)

        if float(w.min()) > eps_null:
            # M is PD
            eta0 = -0.5 * np.linalg.solve(M, s)
            eta_star = eta0
            status = "optimal_pd"
        else:
            # PSD / singular: use pinv + nullspace completion to hit constraint boundary
            M_pinv = V @ np.diag([0.0 if wi <= eps_null else 1.0 / wi for wi in w]) @ V.T
            eta0 = -0.5 * (M_pinv @ s)

            # Nullspace basis
            U0 = V[:, w <= eps_null]
            if U0.size == 0:
                eta_star = eta0
                status = "optimal_psd_no_null_detected"
            else:
                # Choose direction in nullspace that increases objective.
                # Gradient of objective at eta0: grad f = 2A eta0 + b
                grad_f = 2.0 * (A @ eta0) + b
                # Pick u as projection of grad_f onto nullspace (if zero, take any null vector)
                u = U0 @ (U0.T @ grad_f)
                if float(np.linalg.norm(u)) <= 1e-14:
                    u = U0[:, 0]
                u = u / max(1e-14, float(np.linalg.norm(u)))

                # Solve scalar quadratic for alpha so that constraint is active: g(eta0 + alpha u) = r
                # g(eta) = eta^T Ac eta + bc^T eta + cc
                a2 = float(u @ (Ac @ u))
                a1 = float(2.0 * u @ (Ac @ eta0) + bc @ u)
                a0 = float(eta0 @ (Ac @ eta0) + bc @ eta0 + cc - r)

                # Handle degenerate cases
                alphas = []
                if abs(a2) <= 1e-14:
                    if abs(a1) > 1e-14:
                        alphas = [(-a0 / a1)]
                else:
                    disc = a1 * a1 - 4.0 * a2 * a0
                    if disc >= 0.0:
                        sq = float(np.sqrt(disc))
                        alphas = [(-a1 - sq) / (2.0 * a2), (-a1 + sq) / (2.0 * a2)]

                if not alphas:
                    # couldn't hit boundary; return best we have
                    eta_star = eta0
                    status = "psd_nullspace_no_real_boundary_root"
                else:
                    # pick alpha that gives feasible and maximizes objective
                    best_eta = None
                    best_f = -np.inf
                    for alpha in alphas:
                        cand = eta0 + float(alpha) * u
                        if constraint_val(cand) <= r + 1e-6:
                            fv = objective_val(cand)
                            if fv > best_f:
                                best_f = fv
                                best_eta = cand
                    if best_eta is None:
                        eta_star = eta0
                        status = "psd_nullspace_roots_infeasible"
                    else:
                        eta_star = best_eta
                        status = "optimal_psd_with_nullspace_completion"

        primal_val = float(objective_val(eta_star))
        constr_val = float(constraint_val(eta_star))

        # Dual feasibility diagnostics
        dual_feasible = bool(dual_domain_ok(lambda_star) and np.isfinite(dual_val))

        return DualQCQPSolution(
            eta_star=np.asarray(eta_star, dtype=float),
            lambda_star=float(lambda_star),
            primal_value=primal_val,
            dual_value=float(dual_val),
            constraint_value=constr_val,
            feasible=bool(constr_val <= r + 1e-6),
            dual_feasible=dual_feasible,
            status=status,
        )

    def solve_primal_sdp_relaxation(
            self,
            radius: float,
            solver: str | None = None,
            solver_opts: dict | None = None,
            tol: float = 1e-8,
    ) -> DualQCQPSolution:
        import cvxpy as cp
        import numpy as np

        r = float(radius)
        n = self.A.shape[0]

        # Variables (Boyd B.3-style)
        X = cp.Variable((n, n), PSD=True)
        eta = cp.Variable(n)

        # Objective: maximize tr(A X) + b^T eta + c
        obj = cp.Maximize(cp.trace(self.A @ X) + self.b @ eta + self.c)

        # Constraint: tr(Ac X) + bc^T eta + cc <= r
        constr = [cp.trace(self.A_c @ X) + self.b_c @ eta + self.c_c <= r]

        # Lifted LMI: [X eta; eta^T 1] >= 0
        LMI = cp.bmat([
            [X, cp.reshape(eta, (n, 1))],
            [cp.reshape(eta, (1, n)), np.array([[1.0]])]
        ])
        constr.append(LMI >> 0)

        prob = cp.Problem(obj, constr)

        solve_kwargs = {}
        if solver is not None:
            solve_kwargs["solver"] = solver
        if solver_opts:
            solve_kwargs.update(solver_opts)

        try:
            prob.solve(**solve_kwargs)
        except Exception:
            prob.solve(solver=cp.SCS)

        status = str(prob.status)

        if prob.value is None or eta.value is None:
            eta_nan = np.full(n, np.nan)
            return DualQCQPSolution(
                eta_star=eta_nan,
                lambda_star=float("nan"),
                primal_value=float("nan"),
                dual_value=float("nan"),
                constraint_value=float("nan"),
                feasible=False,
                dual_feasible=False,
                status=status,
            )

        eta_star = np.array(eta.value, dtype=float).reshape(-1)

        primal_value = float(eta_star.T @ self.A @ eta_star + self.b.T @ eta_star + self.c)
        constraint_value = float(eta_star.T @ self.A_c @ eta_star + self.b_c.T @ eta_star + self.c_c)

        feasible = constraint_value <= r + 1e3 * tol

        # This is a primal relaxation, so "lambda_star" is not part of the solution.
        # Still fill it with nan.
        # dual_value: prob.value is the SDP relaxation value (>= true primal max, but tight in single-constraint cases)
        dual_value = float(prob.value)

        # Optional: check tightness via rank-1 residual
        Xv = None if X.value is None else np.array(X.value, dtype=float)
        if Xv is not None:
            resid = np.linalg.norm(Xv - np.outer(eta_star, eta_star), ord="fro")
            status = f"{status}; lift_resid={resid:.2e}"

        def objective_val(eta: np.ndarray) -> float:
            eta = eta.reshape(-1)
            return float(eta @ (self.A @ eta) + self.b @ eta + self.c)

        primal_value = objective_val(eta_star)

        return DualQCQPSolution(
            eta_star=eta_star,
            lambda_star=float("nan"),
            primal_value=primal_value,
            dual_value=dual_value,
            constraint_value=float("nan"),
            feasible=feasible,
            dual_feasible=True,
            status=status,
        )
