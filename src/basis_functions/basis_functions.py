from typing import Optional, Literal
from scipy.spatial.distance import pdist, cdist
import numpy as np
from abc import ABC, abstractmethod
import sympy as sp
from scipy.integrate import nquad
from sklearn.cluster import KMeans


class BaseBasisFunction(ABC):
    @abstractmethod
    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """Compute basis features φ(θ)."""
        pass

    @abstractmethod
    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """Compute ∇θ φ(θ)."""
        pass

    def check_C1(self, dim: int) -> bool:
        expr = self.symbolic_expression(dim)
        if expr is None:
            raise NotImplementedError("Symbolic expression not implemented.")
        variables = sp.symbols(f"x0:{dim}")
        try:
            for var in variables:
                sp.diff(expr, var)
            return True
        except Exception:
            return False

    def check_L2(self, dim: int) -> bool:
        def func(*args):
            x = np.array(args)[None, :]  # shape (1, d)
            try:
                val = self.evaluate(x)[0, 0, 0]  # just the first basis function at the first sample
                return float(val ** 2)
            except Exception:
                return np.inf

        try:
            region = [(-10, 10)] * dim
            result, _ = nquad(func, region)
            return np.isfinite(result)
        except Exception:
            return False


class PolynomialBasisFunction(BaseBasisFunction):
    def __init__(self, degree: int):
        self.degree = degree

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        m, d = samples.shape
        values = np.zeros((m, d, self.degree))

        for k in range(1, self.degree + 1):
            values[:, :, k - 1] = samples ** k

        return values  # (m, d, degree)

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        m, d = samples.shape
        grads = []

        for k in range(1, self.degree + 1):
            grad_k = k * samples ** (k - 1)  # (m, d)
            grad_block = np.zeros((m, d, d * self.degree))

            for dim in range(d):
                grad_block[:, dim, dim + d * (k - 1)] = grad_k[:, dim]

            grads.append(grad_block)

        return np.sum(grads, axis=0)  # (m, d, d * degree)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        x = sp.symbols(f"x0:{dim}")
        return sum([x[i] ** self.degree for i in range(dim)])


class RBFBasisFunction(BaseBasisFunction):
    def __init__(
        self,
        posterior_samples: np.ndarray,
        num_basis_functions: int,
        prior_samples: Optional[np.ndarray] = None,
        lengthscale: Optional[float] = None,
        method: Literal["kmeans", "random", "farthest", "kmeans_mix"] = "kmeans",
        estimation_samples_source: Optional[str] = "prior",
        scale_multiplier: float = 1.0,
    ):
        self.rng = np.random.default_rng(None)

        if estimation_samples_source == "prior":
            estimation_samples = prior_samples
        else:
            estimation_samples = posterior_samples

        self.centers = self._select_centers(
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            num_centers=num_basis_functions,
            method=method,
        )

        if lengthscale is None:
            self.lengthscale = self._estimate_lengthscale(
                centers=self.centers,
                samples=estimation_samples,
                multiplier=scale_multiplier,
            )
        else:
            self.lengthscale = float(lengthscale)

    def _select_centers(
        self,
        posterior_samples: np.ndarray,
        prior_samples: Optional[np.ndarray],
        num_centers: int,
        method: str,
    ) -> np.ndarray:
        if prior_samples is not None:
            X = np.asarray(prior_samples, dtype=float)
        else:
            X = np.asarray(posterior_samples, dtype=float)

        m, d = X.shape

        if method == "kmeans":
            n = min(num_centers, m)
            return KMeans(n_clusters=n, random_state=0).fit(X).cluster_centers_

        if method == "random":
            n = min(num_centers, m)
            idx = self.rng.choice(m, n, replace=False)
            return X[idx]

        if method == "farthest":
            return self._farthest_point_sampling(X, min(num_centers, m))

        if method == "kmeans_mix":
            Xp = np.asarray(prior_samples, dtype=float)
            Xpost = np.asarray(posterior_samples, dtype=float)
            m_post = min(len(Xpost), max(1, num_centers * 50))
            m_prior = min(len(Xp), max(1, num_centers * 50))
            Xmix = np.vstack([
                Xpost[self.rng.choice(len(Xpost), m_post, replace=False)],
                Xp[self.rng.choice(len(Xp), m_prior, replace=False)],
            ])

            return KMeans(n_clusters=min(num_centers, len(Xmix)), random_state=0).fit(Xmix).cluster_centers_

        raise ValueError(f"Unknown center selection method: {method}")

    def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
        n = X.shape[0]
        if k <= 0:
            return np.empty((0, X.shape[1]))
        if k == 1:
            return X[[self.rng.integers(0, n)]]
        idx0 = int(self.rng.integers(0, n))
        centers_idx = [idx0]
        dist = cdist(X[[idx0]], X).reshape(-1)
        for _ in range(1, k):
            i = int(np.argmax(dist))
            centers_idx.append(i)
            dist = np.minimum(dist, cdist(X[[i]], X).reshape(-1))
        return X[np.array(centers_idx)]

    def _estimate_lengthscale(self, centers: np.ndarray, samples: np.ndarray,
                              source: str = "samples", multiplier: float = 1.0,
                              floor_frac: float = 0.1) -> float:
        if centers.shape[0] < 2:
            return 1.0
        if source == "samples":
            m = np.median(pdist(samples.reshape(-1, samples.shape[-1])))
        else:
            m = np.median(pdist(centers))

        ell = multiplier * m
        floor = floor_frac * np.std(samples, axis=0).mean()
        ls = float(max(ell, floor))
        print(f"Selected lengthscale for RBF basis function: {ls}.")

        return ls

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        diffs = samples[:, None, :] - self.centers[None, :, :]
        squared = diffs ** 2
        squared = np.transpose(squared, (0, 2, 1))

        return np.exp(-squared / (2 * self.lengthscale ** 2))  # (m, d, B)

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        diffs = samples[:, None, :] - self.centers[None, :, :]  # (m, B, d)
        rbf_vals = np.exp(-np.sum(diffs ** 2, axis=-1) / (2 * self.lengthscale ** 2))  # (m, B)
        grad = (-1 / self.lengthscale ** 2) * (diffs * rbf_vals[:, :, None])  # (m, B, d)

        return np.transpose(grad, (0, 2, 1))  # (m, d, B)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        x = sp.symbols(f"x0:{dim}")
        c = self.centers[0]  # just validate the first center
        return sp.exp(-sum([(x[i] - c[i])**2 for i in range(dim)]) / (2 * self.lengthscale**2))


class SigmoidBasisFunction(BaseBasisFunction):
    def __init__(
        self,
        posterior_samples: np.ndarray,
        num_basis_functions: int,
        prior_samples: Optional[np.ndarray] = None,
        scale: Optional[float] = None,
        method: Literal["kmeans", "random", "farthest", "kmeans_mix"] = "kmeans",
        estimation_samples_source: Optional[str] = "prior",
        scale_multiplier: float = 1.0,
    ):
        self.rng = np.random.default_rng(None)

        if estimation_samples_source == "prior":
            estimation_samples = prior_samples
        else:
            estimation_samples = posterior_samples

        self.centers = self._select_centers(
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            num_centers=num_basis_functions,
            method=method,
        )

        if scale is None:
            self.scale = self._estimate_scale(
                centers=self.centers,
                samples=estimation_samples,
                multiplier=scale_multiplier,
            )
            print(f"Selected scale for Sigmoid basis function: {self.scale}.")
        else:
            self.scale = float(scale)

    def _select_centers(
        self,
        posterior_samples: np.ndarray,
        prior_samples: Optional[np.ndarray],
        num_centers: int,
        method: str,
    ) -> np.ndarray:
        if prior_samples is not None:
            X = np.asarray(prior_samples, dtype=float)
        else:
            X = np.asarray(posterior_samples, dtype=float)

        m, d = X.shape

        if method == "kmeans":
            n = min(num_centers, m)
            return KMeans(n_clusters=n, random_state=0).fit(X).cluster_centers_

        if method == "random":
            n = min(num_centers, m)
            idx = self.rng.choice(m, n, replace=False)
            return X[idx]

        if method == "farthest":
            return self._farthest_point_sampling(X, min(num_centers, m))

        if method == "kmeans_mix":
            Xp = np.asarray(prior_samples, dtype=float)
            Xpost = np.asarray(posterior_samples, dtype=float)
            m_post = min(len(Xpost), max(1, num_centers * 50))
            m_prior = min(len(Xp), max(1, num_centers * 50))
            Xmix = np.vstack([
                Xpost[self.rng.choice(len(Xpost), m_post, replace=False)],
                Xp[self.rng.choice(len(Xp), m_prior, replace=False)],
            ])

            return KMeans(n_clusters=min(num_centers, len(Xmix)), random_state=0).fit(Xmix).cluster_centers_

        raise ValueError(f"Unknown center selection method: {method}")

    def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
        n = X.shape[0]
        if k <= 0:
            return np.empty((0, X.shape[1]))
        if k == 1:
            return X[[self.rng.integers(0, n)]]
        idx0 = int(self.rng.integers(0, n))
        centers_idx = [idx0]
        dist = cdist(X[[idx0]], X).reshape(-1)
        for _ in range(1, k):
            i = int(np.argmax(dist))
            centers_idx.append(i)
            dist = np.minimum(dist, cdist(X[[i]], X).reshape(-1))
        return X[np.array(centers_idx)]

    def _estimate_scale(
        self,
        centers: np.ndarray,
        samples: np.ndarray,
        *,
        source: str = "samples",
        multiplier: float = 0.01,
        floor_frac: float = 0.1,
    ) -> float:
        if centers.shape[0] < 2:
            return 1.0
        if source == "samples":
            m = np.median(pdist(samples.reshape(-1, samples.shape[-1])))
        else:
            m = np.median(pdist(centers))
        s = multiplier * m
        floor = floor_frac * np.std(samples, axis=0).mean()
        return float(max(s, floor))

    @staticmethod
    def _sigmoid(u: np.ndarray) -> np.ndarray:
        # stable logistic
        return 1.0 / (1.0 + np.exp(-u))

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """
        Returns φ(x) with shape (m, d, B),
        where the last axis indexes basis functions.
        """
        # diffs: (m, B, d)
        diffs = samples[:, None, :] - self.centers[None, :, :]
        u = diffs / self.scale                       # (m, B, d)
        vals = self._sigmoid(u)                      # (m, B, d)
        return np.transpose(vals, (0, 2, 1))         # (m, d, B)

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """
        ∂φ/∂x: shape (m, d, B).
        d/dx σ((x-c)/s) = (1/s) * σ(u)*(1-σ(u)) component-wise.
        """
        diffs = samples[:, None, :] - self.centers[None, :, :]  # (m, B, d)
        u = diffs / self.scale                                  # (m, B, d)
        sig = self._sigmoid(u)
        d_sig_du = sig * (1.0 - sig)                            # (m, B, d)
        grad_mBd = (1.0 / self.scale) * d_sig_du                # (m, B, d)
        return np.transpose(grad_mBd, (0, 2, 1))                # (m, d, B)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        x = sp.symbols(f"x0:{dim}")
        c = self.centers[0]
        s = sp.Float(self.scale)

        return 1 / (1 + sp.exp(-(sum((x[i] - c[i]) for i in range(dim)) / s)))


# class RBFBasisFunctionMultidim(BaseBasisFunction):
#     def __init__(
#         self,
#         posterior_samples: np.ndarray,
#         num_basis_functions: int,
#         prior_samples: Optional[np.ndarray] = None,
#         lengthscale: Optional[np.ndarray] = None,  # per-dim (d,)
#         method: Literal["kmeans", "random", "farthest", "kmeans_mix"] = "kmeans",
#         estimation_samples_source: Optional[str] = "prior",
#         scale_multiplier: float = 1.0,
#         floor_frac: float = 0.1,
#     ):
#         """
#         Multidim RBF basis with per-dimension lengthscales l \in R^d (vector).
#
#         Outputs:
#           evaluate(samples) -> (m, d, B):  phi_{i,b}(x) = exp(-(x_i - c_{b,i})^2 / (2 l_i^2))
#           gradient(samples) -> (m, d, B):  d/dx_i phi_{i,b}(x) = -(x_i - c_{b,i})/l_i^2 * phi_{i,b}(x)
#         """
#         self.rng = np.random.default_rng(None)
#
#         if estimation_samples_source == "prior":
#             estimation_samples = prior_samples
#         else:
#             estimation_samples = posterior_samples
#
#         # Choose centers (B, d)
#         self.centers = self._select_centers(
#             posterior_samples=posterior_samples,
#             prior_samples=prior_samples,
#             num_centers=num_basis_functions,
#             method=method,
#         )
#
#         # Per-dimension lengthscale (d,)
#         if lengthscale is None:
#             if estimation_samples is None:
#                 # fallback: estimate from centers if samples not provided
#                 ls = self._estimate_lengthscale_vector_from_centers(
#                     centers=self.centers,
#                     multiplier=scale_multiplier,
#                     floor_frac=floor_frac,
#                 )
#             else:
#                 ls = self._estimate_lengthscale_vector_from_samples(
#                     samples=estimation_samples,
#                     multiplier=scale_multiplier,
#                     floor_frac=floor_frac,
#                 )
#             self.lengthscale = ls
#         else:
#             ls = np.asarray(lengthscale, dtype=float)
#             if ls.ndim != 1:
#                 raise ValueError("lengthscale must be a 1D array of shape (d,).")
#             self.lengthscale = ls
#
#         # Nice printout
#         ls_str = np.array2string(self.lengthscale, precision=4, separator=", ")
#         print(f"Selected per-dimension lengthscales for RBF basis: {ls_str}")
#
#     # ---------- center selection (same strategies as before) ----------
#     def _select_centers(
#         self,
#         posterior_samples: np.ndarray,
#         prior_samples: Optional[np.ndarray],
#         num_centers: int,
#         method: str,
#     ) -> np.ndarray:
#         if prior_samples is not None:
#             X = np.asarray(prior_samples, dtype=float)
#         else:
#             X = np.asarray(posterior_samples, dtype=float)
#
#         m, d = X.shape
#
#         if method == "kmeans":
#             n = min(num_centers, m)
#             return KMeans(n_clusters=n, random_state=0).fit(X).cluster_centers_
#
#         if method == "random":
#             n = min(num_centers, m)
#             idx = self.rng.choice(m, n, replace=False)
#             return X[idx]
#
#         if method == "farthest":
#             return self._farthest_point_sampling(X, min(num_centers, m))
#
#         if method == "kmeans_mix":
#             if prior_samples is None:
#                 raise ValueError("kmeans_mix requires prior_samples.")
#             Xp = np.asarray(prior_samples, dtype=float)
#             Xpost = np.asarray(posterior_samples, dtype=float)
#             m_post = min(len(Xpost), max(1, num_centers * 50))
#             m_prior = min(len(Xp), max(1, num_centers * 50))
#             Xmix = np.vstack([
#                 Xpost[self.rng.choice(len(Xpost), m_post, replace=False)],
#                 Xp[self.rng.choice(len(Xp), m_prior, replace=False)],
#             ])
#             return KMeans(n_clusters=min(num_centers, len(Xmix)), random_state=0).fit(Xmix).cluster_centers_
#
#         raise ValueError(f"Unknown center selection method: {method}")
#
#     def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
#         n = X.shape[0]
#         if k <= 0:
#             return np.empty((0, X.shape[1]))
#         if k == 1:
#             return X[[self.rng.integers(0, n)]]
#         idx0 = int(self.rng.integers(0, n))
#         centers_idx = [idx0]
#         dist = cdist(X[[idx0]], X).reshape(-1)
#         for _ in range(1, k):
#             i = int(np.argmax(dist))
#             centers_idx.append(i)
#             dist = np.minimum(dist, cdist(X[[i]], X).reshape(-1))
#         return X[np.array(centers_idx)]
#
#     def _median_heuristic_per_dim(self, x: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
#         x = np.asarray(x, dtype=float)
#         if x.ndim != 2:
#             raise ValueError("reference_data must be a 2D array of shape (n, d).")
#         n, d = x.shape
#         if n < 2:
#             return np.sqrt(np.var(x, axis=0) + jitter)
#
#         diffs = x[:, None, :] - x[None, :, :]  # (n, n, d)
#         iu = np.triu_indices(n, k=1)
#         diffs = diffs[iu]                       # (n*(n-1)/2, d)
#         med_sq = np.median(diffs**2, axis=0)    # (d,)
#         return np.sqrt(med_sq + jitter)         # (d,)
#
#     def _estimate_lengthscale_vector_from_samples(
#         self,
#         samples: np.ndarray,
#         multiplier: float = 1.0,
#         floor_frac: float = 0.1,
#     ) -> np.ndarray:
#         ls = self._median_heuristic_per_dim(samples)
#         floor = floor_frac * np.std(samples, axis=0)
#         ls = np.maximum(multiplier * ls, floor)
#         return ls.astype(float)
#
#     def _estimate_lengthscale_vector_from_centers(
#         self,
#         centers: np.ndarray,
#         multiplier: float = 1.0,
#         floor_frac: float = 0.1,
#     ) -> np.ndarray:
#         ls = self._median_heuristic_per_dim(centers)
#         floor = floor_frac * np.std(centers, axis=0)
#         ls = np.maximum(multiplier * ls, floor)
#         return ls.astype(float)
#
#     # ---------- basis & gradient ----------
#     def evaluate(self, samples: np.ndarray) -> np.ndarray:
#         """
#         Return per-dimension basis values:
#           phi_{i,b}(x) = exp( - (x_i - c_{b,i})^2 / (2 l_i^2) )
#         Shape: (m, d, B)
#         """
#         samples = np.asarray(samples, dtype=float)
#         # diffs: (m, B, d)
#         diffs = samples[:, None, :] - self.centers[None, :, :]
#         # (m, B, d) / (d,)
#         denom = 2.0 * (self.lengthscale**2)  # (d,)
#         vals = np.exp(-(diffs**2) / denom)   # (m, B, d)
#         return np.transpose(vals, (0, 2, 1)) # (m, d, B)
#
#     def gradient(self, samples: np.ndarray) -> np.ndarray:
#         """
#         Per-dimension gradient of the per-dim basis:
#           d/dx_i phi_{i,b}(x) = -(x_i - c_{b,i}) / l_i^2 * phi_{i,b}(x)
#         Shape: (m, d, B)
#         """
#         samples = np.asarray(samples, dtype=float)
#         diffs = samples[:, None, :] - self.centers[None, :, :]  # (m, B, d)
#         denom = 2.0 * (self.lengthscale**2)                     # (d,)
#         phi = np.exp(-(diffs**2) / denom)                       # (m, B, d)
#         # factor: (m, B, d) with division by l_i^2
#         factor = -diffs / (self.lengthscale**2)                 # (m, B, d)
#         grad = factor * phi                                     # (m, B, d)
#         return np.transpose(grad, (0, 2, 1))                    # (m, d, B)
#
#     def symbolic_expression(self, dim: int) -> sp.Expr:
#         """
#         Returns a separable per-dimension RBF expression for the FIRST center b=0:
#           exp( - (x_i - c_{0,i})^2 / (2 l_i^2) )  (for a given i)
#         """
#         x = sp.symbols(f"x0:{dim}")
#         c = self.centers[0]
#         exprs = [
#             sp.exp(-((x[i] - c[i])**2) / (2 * (self.lengthscale[i]**2)))
#             for i in range(dim)
#         ]
#         # return a tuple-like additive or list — here we return a vector symbolic form
#         return sp.Tuple(*exprs)

import numpy as np
import sympy as sp
from typing import Optional, Literal
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from numpy.linalg import eigh

class RBFBasisFunctionMultidim(BaseBasisFunction):
    def __init__(
        self,
        posterior_samples: np.ndarray,
        num_basis_functions: int,
        prior_samples: Optional[np.ndarray] = None,
        # Lengthscale controls:
        # - metric="diag": lengthscale is a vector (d,)
        # - metric="full": precision is a PD matrix (d,d) (inverse covariance scaled)
        lengthscale: Optional[np.ndarray] = None,        # used when metric="diag" (per-dim)
        precision: Optional[np.ndarray] = None,          # used when metric="full" (inverse covariance-like)
        metric: Literal["diag", "full"] = "diag",
        # Center selection:
        method: Literal["kmeans", "random", "farthest", "kmeans_mix"] = "kmeans",
        # Which samples to use for bandwidth estimation:
        estimation_samples_source: Optional[str] = "prior",  # "prior" or "posterior"
        # Scaling / regularization:
        scale_multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ):
        """
        Multidim RBF basis with either per-dimension (diag) or full (non-diagonal) bandwidth.

        metric="diag":
          phi_{i,b}(x) = exp( - (x_i - c_{b,i})^2 / (2 l_i^2) ), per-dimension separable basis.
          evaluate(samples) -> (m, d, B)
          gradient(samples) -> (m, d, B) with per-dim formula.

        metric="full":
          phi_b(x)       = exp( -1/2 (x - c_b)^T P (x - c_b) ), with shared precision matrix P ≻ 0.
          evaluate       -> (m, d, B) by repeating the scalar across d to keep shape compatibility.
          gradient       -> (m, d, B) using full-matrix derivative: -P(x-c_b) * phi_b(x)
        """
        self.rng = np.random.default_rng(None)
        self.metric = metric

        # Pick samples that control the bandwidth/precision estimation
        if estimation_samples_source == "prior":
            estimation_samples = prior_samples
        else:
            estimation_samples = posterior_samples

        # Choose centers (B, d)
        self.centers = self._select_centers(
            posterior_samples=posterior_samples,
            prior_samples=prior_samples,
            num_centers=num_basis_functions,
            method=method,
        )
        _, self.dim = self.centers.shape
        self.num_basis = num_basis_functions

        # Estimate bandwidth structure
        if self.metric == "diag":
            if lengthscale is None:
                if estimation_samples is None:
                    ls = self._estimate_lengthscale_vector_from_centers(
                        centers=self.centers,
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                    )
                else:
                    ls = self._estimate_lengthscale_vector_from_samples(
                        samples=np.asarray(estimation_samples, dtype=float),
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                    )
            else:
                ls = np.asarray(lengthscale, dtype=float)
                if ls.ndim != 1 or ls.shape[0] != self.dim:
                    raise ValueError(f"lengthscale must be shape (d,), got {ls.shape}.")
            self.lengthscale = ls
            self.precision = None  # not used in diag mode
            ls_str = np.array2string(self.lengthscale, precision=4, separator=", ")
            print(f"[RBF diag] lengthscales l: {ls_str}")

        elif self.metric == "full":
            if precision is None:
                if estimation_samples is None:
                    P = self._estimate_precision_from_centers(
                        centers=self.centers,
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                        jitter=jitter,
                    )
                else:
                    P = self._estimate_precision_from_samples(
                        samples=np.asarray(estimation_samples, dtype=float),
                        multiplier=scale_multiplier,
                        floor_frac=floor_frac,
                        jitter=jitter,
                    )
            else:
                P = np.asarray(precision, dtype=float)
                if P.shape != (self.dim, self.dim):
                    raise ValueError(f"precision must be (d,d), got {P.shape}.")
                # ensure symmetric PD
                P = 0.5 * (P + P.T)
                # very small regularization to avoid numerical issues
                w, V = eigh(P)
                w = np.maximum(w, jitter)
                P = (V * w) @ V.T
            self.precision = P
            self.lengthscale = None  # not used in full mode
            # Print eigvals for sanity
            w, _ = eigh(self.precision)
            w_str = np.array2string(w, precision=4, separator=", ")
            print(f"[RBF full] precision: {self.precision}")
            print(f"[RBF full] precision eigvals: {w_str}")
        else:
            raise ValueError("metric must be 'diag' or 'full'.")

    # ---------- center selection (same strategies as before) ----------
    def _select_centers(
        self,
        posterior_samples: np.ndarray,
        prior_samples: Optional[np.ndarray],
        num_centers: int,
        method: str,
    ) -> np.ndarray:
        if prior_samples is not None:
            X = np.asarray(prior_samples, dtype=float)
        else:
            X = np.asarray(posterior_samples, dtype=float)

        m, d = X.shape

        if method == "kmeans":
            n = min(num_centers, m)
            return KMeans(n_clusters=n, random_state=0).fit(X).cluster_centers_

        if method == "random":
            n = min(num_centers, m)
            idx = self.rng.choice(m, n, replace=False)
            return X[idx]

        if method == "farthest":
            return self._farthest_point_sampling(X, min(num_centers, m))

        if method == "kmeans_mix":
            if prior_samples is None:
                raise ValueError("kmeans_mix requires prior_samples.")
            Xp = np.asarray(prior_samples, dtype=float)
            Xpost = np.asarray(posterior_samples, dtype=float)
            m_post = min(len(Xpost), max(1, num_centers * 50))
            m_prior = min(len(Xp), max(1, num_centers * 50))
            Xmix = np.vstack([
                Xpost[self.rng.choice(len(Xpost), m_post, replace=False)],
                Xp[self.rng.choice(len(Xp), m_prior, replace=False)],
            ])
            return KMeans(n_clusters=min(num_centers, len(Xmix)), random_state=0).fit(Xmix).cluster_centers_

        raise ValueError(f"Unknown center selection method: {method}")

    def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
        n = X.shape[0]
        if k <= 0:
            return np.empty((0, X.shape[1]))
        if k == 1:
            return X[[self.rng.integers(0, n)]]
        idx0 = int(self.rng.integers(0, n))
        centers_idx = [idx0]
        dist = cdist(X[[idx0]], X).reshape(-1)
        for _ in range(1, k):
            i = int(np.argmax(dist))
            centers_idx.append(i)
            dist = np.minimum(dist, cdist(X[[i]], X).reshape(-1))
        return X[np.array(centers_idx)]

    # ---------- diagonal (per-dim) lengthscale estimation ----------
    def _median_heuristic_per_dim(self, x: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
        """
        Per-dimension median heuristic:
            l_i = sqrt(median_{i<j} (x_i - x_j)^2 + jitter)
        x: (n, d)
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("reference_data must be a 2D array of shape (n, d).")
        n, d = x.shape
        if n < 2:
            return np.sqrt(np.var(x, axis=0) + jitter)

        diffs = x[:, None, :] - x[None, :, :]  # (n, n, d)
        iu = np.triu_indices(n, k=1)
        diffs = diffs[iu]                       # (n*(n-1)/2, d)
        med_sq = np.median(diffs**2, axis=0)    # (d,)
        return np.sqrt(med_sq + jitter)         # (d,)

    def _estimate_lengthscale_vector_from_samples(
        self,
        samples: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
    ) -> np.ndarray:
        ls = self._median_heuristic_per_dim(samples)
        floor = floor_frac * np.std(samples, axis=0)
        ls = np.maximum(multiplier * ls, floor)
        return ls.astype(float)

    def _estimate_lengthscale_vector_from_centers(
        self,
        centers: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
    ) -> np.ndarray:
        ls = self._median_heuristic_per_dim(centers)
        floor = floor_frac * np.std(centers, axis=0)
        ls = np.maximum(multiplier * ls, floor)
        return ls.astype(float)

    # ---------- full (non-diagonal) precision estimation ----------
    def _estimate_precision_from_samples(
        self,
        samples: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ) -> np.ndarray:
        """
        Estimate a shared precision matrix P ≻ 0:
          - compute covariance Σ from samples
          - regularize eigenvalues (floor_frac * mean_eig + jitter)
          - invert and scale: P = (1 / multiplier^2) * Σ^{-1}
        """
        X = np.asarray(samples, dtype=float)
        Σ = np.cov(X, rowvar=False)  # (d,d)
        # symmetrize for safety
        Σ = 0.5 * (Σ + Σ.T)
        # eigen-regularize
        w, V = eigh(Σ)
        mean_eig = float(np.mean(np.maximum(w, 0.0)))
        w_reg = np.maximum(w, floor_frac * mean_eig + jitter)
        Σ_reg = (V * w_reg) @ V.T
        # invert (SPD, so safe)
        w_inv = 1.0 / w_reg
        Σ_inv = (V * w_inv) @ V.T
        # scale by multiplier (acts like global lengthscale)
        P = Σ_inv / (multiplier**2)
        # final symmetrize
        return 0.5 * (P + P.T)

    def _estimate_precision_from_centers(
        self,
        centers: np.ndarray,
        multiplier: float = 1.0,
        floor_frac: float = 0.1,
        jitter: float = 1e-8,
    ) -> np.ndarray:
        return self._estimate_precision_from_samples(
            samples=centers,
            multiplier=multiplier,
            floor_frac=floor_frac,
            jitter=jitter,
        )

    # ---------- basis & gradient ----------
    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """
        metric="diag":
            phi_{i,b}(x) = exp( - (x_i - c_{b,i})^2 / (2 l_i^2) )
            return shape: (m, d, B)
        metric="full":
            phi_b(x)     = exp( -1/2 (x - c_b)^T P (x - c_b) ), shared P
            return shape: (m, d, B) with the scalar repeated along d to match downstream API
        """
        X = np.asarray(samples, dtype=float)                # (m, d)
        m = X.shape[0]
        B = self.num_basis
        d = self.dim

        if self.metric == "diag":
            # diffs: (m, B, d)
            diffs = X[:, None, :] - self.centers[None, :, :]
            denom = 2.0 * (self.lengthscale**2)[None, None, :]  # (1,1,d)
            vals = np.exp(-(diffs**2) / denom)                  # (m, B, d)
            return np.transpose(vals, (0, 2, 1))                # (m, d, B)

        # metric == "full"
        # diffs: (m, B, d)
        diffs = X[:, None, :] - self.centers[None, :, :]
        # r^2: (m, B) via quadratic form with shared precision P
        # einsum: (m,B,d) * (d,d) * (m,B,d) -> (m,B)
        r2 = np.einsum("mbi,ij,mbj->mb", diffs, self.precision, diffs, optimize=True)
        phi = np.exp(-0.5 * r2)                              # (m, B)
        # repeat along d to keep (m,d,B) interface
        phi_rep = np.repeat(phi[:, None, :], d, axis=1)      # (m, d, B)
        return phi_rep

    def gradient(self, samples: np.ndarray) -> np.ndarray:
        """
        metric="diag":
            d/dx_i phi_{i,b}(x) = -(x_i - c_{b,i}) / l_i^2 * phi_{i,b}(x)
        metric="full":
            ∇_x phi_b(x) = -P (x - c_b) * phi_b(x)
        Returns shape (m, d, B)
        """
        X = np.asarray(samples, dtype=float)                # (m, d)
        m = X.shape[0]
        B = self.num_basis
        d = self.dim

        if self.metric == "diag":
            diffs = X[:, None, :] - self.centers[None, :, :]                 # (m, B, d)
            denom = 2.0 * (self.lengthscale**2)[None, None, :]               # (1,1,d)
            phi = np.exp(-(diffs**2) / denom)                                # (m, B, d)
            factor = -diffs / (self.lengthscale**2)[None, None, :]           # (m, B, d)
            grad = factor * phi                                              # (m, B, d)
            return np.transpose(grad, (0, 2, 1))                             # (m, d, B)

        # metric == "full"
        diffs = X[:, None, :] - self.centers[None, :, :]                     # (m, B, d)
        # compute P (x - c_b) for all b: (d,d) @ (m,B,d)^T -> (m,B,d)
        Px = np.einsum("ij,mbj->mbi", self.precision, diffs, optimize=True)  # (m, B, d)
        r2 = np.einsum("mbi,ij,mbj->mb", diffs, self.precision, diffs, optimize=True)  # (m,B)
        phi = np.exp(-0.5 * r2)                                              # (m, B)
        grad = -Px * phi[:, :, None]                                         # (m, B, d)
        return np.transpose(grad, (0, 2, 1))                                 # (m, d, B)

    def symbolic_expression(self, dim: int) -> sp.Expr:
        """
        Returns a symbolic form for the FIRST center b=0.
        metric="diag": vector of per-dim terms exp( - (x_i - c_i)^2 / (2 l_i^2) )
        metric="full": scalar exp( -1/2 (x - c)^T P (x - c) ) replicated conceptually across dims
        """
        x = sp.symbols(f"x0:{dim}")
        c = self.centers[0]
        if self.metric == "diag":
            exprs = [
                sp.exp(-((x[i] - c[i])**2) / (2 * (self.lengthscale[i]**2)))
                for i in range(dim)
            ]
            return sp.Tuple(*exprs)
        # full
        P = sp.Matrix(self.precision)
        xm = sp.Matrix(x)
        cm = sp.Matrix(c)
        quad = (xm - cm).T * P * (xm - cm)
        expr = sp.exp(-sp.Rational(1, 2) * quad[0])
        # return as a tuple of identical entries to reflect (d,)-replication in evaluate()
        return sp.Tuple(*[expr for _ in range(dim)])


