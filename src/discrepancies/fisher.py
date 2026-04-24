import numpy as np
from typing import Callable


class FisherDivergence:
    """
    Computes the Fisher Divergence using a V-statistic formulation.

    Parameters
    ----------
    score_fn : Callable[[np.ndarray], np.ndarray]
        Function computing the Stein score vector ∇ log p(x).
        Should return array of shape (m, d).
    """

    def __init__(
        self,
        scores_ref: np.ndarray,
        score_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.scores_ref = scores_ref
        self.score_fn = score_fn

    def compute(self, samples: np.ndarray) -> float:
        """
        Computes the Fisher Divergence using a V-statistic formulation.

        Parameters
        ----------
        samples : np.ndarray
            Samples of shape (m, d).

        Returns
        -------
        float
            The estimated KSD^2 value.
        """
        m, d = samples.shape
        scores = self.score_fn(samples)  # shape (m, d)

        term1 = self._compute_squared_term(self.scores_ref)
        term2 = self._compute_squared_term(scores)
        term3 = self._compute_cross_term(self.scores_ref, scores)
        val = np.sum(term1 + term2) / (2*m) - np.sum(term3)/m

        if val < 0.0:
            raise ValueError("The Fisher Divergence estimation is negative and failed.")
        else:
            return val

    @staticmethod
    def _compute_squared_term(scores: np.ndarray) -> np.ndarray:
        """Compute ∑ s(x_i)^T s(x_j)"""
        return np.einsum('ik,jk->ij', scores, scores)

    @staticmethod
    def _compute_cross_term(scores1: np.ndarray, scores2: np.ndarray) -> np.ndarray:
        """Compute ∑ s(x_i)^T s(x_j)"""
        return np.einsum('ik,jk->ij', scores1, scores2)


class FisherDivergenceBase:
    """
    Computes the Fisher Divergence using a V-statistic formulation.
    """

    def __init__(
        self,
        scores: np.ndarray,
        scores_ref: np.ndarray,
    ) -> None:
        self.scores = scores
        self.scores_ref = scores_ref

    def compute(self) -> float:
        diff = self.scores_ref - self.scores
        val = float(np.mean(np.sum(diff * diff, axis=1)))
        if val < 0.0:
            raise ValueError("The Fisher Divergence estimation is negative and failed.")
        return val

    @staticmethod
    def _compute_squared_term(scores: np.ndarray) -> np.ndarray:
        """Compute ∑ s(x_i)^T s(x_j)"""
        return np.einsum('ik,jk->ij', scores, scores)

    @staticmethod
    def _compute_cross_term(scores1: np.ndarray, scores2: np.ndarray) -> np.ndarray:
        """Compute ∑ s(x_i)^T s(x_j)"""
        return np.einsum('ik,jk->ij', scores1, scores2)
