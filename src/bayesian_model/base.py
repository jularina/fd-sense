from __future__ import annotations
from typing import Callable

from utils.utils import ArrayLike

class BayesianModel:
    def __init__(
        self,
        prior_score: Callable[[ArrayLike], ArrayLike],
        loss_score: Callable[[ArrayLike], ArrayLike],
    ):
        """
        Bayesian model with score functions for prior and loss.

        Args:
            prior_score: Function s_pi(theta): R^D -> R^D
            loss_score: Function s_l,beta(theta): R^D -> R^D
        """
        self.prior_score = prior_score
        self.loss_score = loss_score

    def posterior_score(self, theta: ArrayLike) -> ArrayLike:
        return self.prior_score(theta) + self.loss_score(theta)