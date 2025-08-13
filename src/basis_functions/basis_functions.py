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
