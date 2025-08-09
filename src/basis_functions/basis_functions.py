from typing import Optional
from scipy.spatial.distance import pdist
import numpy as np
from abc import ABC, abstractmethod
import sympy as sp
from scipy.integrate import nquad


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
        samples: np.ndarray,
        num_basis_functions: int,
        lengthscale: Optional[float] = None,
        method: str = "kmeans",  # or "random"
    ):
        self.centers = self._select_centers(samples, num_basis_functions, method)

        if lengthscale is None:
            self.lengthscale = self._estimate_lengthscale(centers=self.centers, samples=samples)
        else:
            self.lengthscale = lengthscale

    def _select_centers(self, samples: np.ndarray, num_centers: int, method: str) -> np.ndarray:
        if method == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(samples)
            return kmeans.cluster_centers_
        elif method == "random":
            idx = np.random.choice(len(samples), num_centers, replace=False)
            return samples[idx]
        else:
            raise ValueError(f"Unknown center selection method: {method}")

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
        return float(max(ell, floor))

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
