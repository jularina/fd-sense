from abc import ABC, abstractmethod
from typing import Any, Dict, Type
import numpy as np

from src.utils.typing import ArrayLike


class BayesianModel(ABC):
    def __init__(self, data_config: Any):
        """
        Base class for Bayesian models.
        """
        self.true_dgp = data_config.true_dgp
        self.observations_num: int = data_config.observations_num
        self.observations: np.ndarray = self.true_dgp.sample(self.observations_num)
        self.x_bar: np.ndarray = np.mean(self.observations, axis=0)
        self.loss_lr: float = data_config.loss_lr
        self.loss: Any = data_config.loss
        self.prior: Any = data_config.base_prior
        self.prior_init: Any = data_config.base_prior
        self.loss_lr_init: float = data_config.loss_lr
        self.m: int = data_config.posterior_samples_num

    @abstractmethod
    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Draw samples from the posterior distribution.
        """
        raise NotImplementedError

    def sample_from_base_prior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Draw samples from the posterior distribution.
        """
        return self.prior_init.sample(n_samples)

    def set_prior_parameters(self, params: Dict[str, Any], distribution_cls: Type) -> None:
        """
        Set or update the prior distribution.

        Args:
            params: Dictionary of prior parameters.
            distribution_cls: Distribution class to instantiate the prior.
        """
        self.prior = distribution_cls(**params)

    def set_lr_parameter(self, lr: float) -> None:
        """
        Set or update the loss function parameters.

        Args:
            lr: Learning rate for the loss term.
        """
        self.loss_lr = lr

    def prior_score(self, x: ArrayLike) -> np.ndarray:
        """Compute gradient of log prior."""
        return self.prior.grad_log_pdf(x)

    def loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        """Compute gradient of log likelihood (scaled by learning rate)."""
        grad = self.loss.grad_log_pdf(x, self.x_bar, self.observations_num)
        return self.loss_lr * grad if multiply_by_lr else grad

    def posterior_score(self, x: ArrayLike) -> np.ndarray:
        """Compute posterior score (prior + likelihood)."""
        prior_grad = self.prior_score(x)
        loss_grad = self.loss_score(x)
        return prior_grad + loss_grad

    def jacobian_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """Return Jacobian of sufficient statistics."""
        return self.prior.grad_sufficient_statistics(x)

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        """Gradient of log base measure."""
        return self.prior.grad_log_base_measure(x)
