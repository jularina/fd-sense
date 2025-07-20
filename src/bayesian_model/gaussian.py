import numpy as np
from typing import Dict, Any
import pymc as pm
from hydra.utils import instantiate

from src.bayesian_model.base import BayesianModel
from src.distributions.gaussian import Gaussian
from src.utils.typing import ArrayLike
from utils.instantiation import get_class_from_path


class SimpleGaussianModel(BayesianModel):
    """
    A simple Bayesian model for univariate Gaussian likelihood with Gaussian prior.
    Assumes data ~ Normal(mu, sigma_obs^2), and prior ~ Normal(mu0, sigma0^2).
    """

    def __init__(self, data_config):
        # Instantiate true data generator and sample observations
        self.true_dgp = data_config.true_dgp
        self.observations_num = data_config.observations_num
        self.observations = self.true_dgp.sample(self.observations_num)
        # self.observations = np.full((self.observations_num, 1), 3.0)
        self.x_bar = np.mean(self.observations)
        self.loss_lr = data_config.loss_lr

        # Prior config
        self.prior = data_config.base_prior

        # Likelihood config
        self.loss = data_config.loss

    def set_prior_parameters(self, params: Dict[str, float]) -> None:
        """Update prior parameters for optimization.

        Args:
            params (Dict[str, float]): Dictionary of new prior parameters.
        """
        self.prior = Gaussian(**params)

    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Sample from the closed-form conjugate posterior of a Gaussian model.

        Returns:
            np.ndarray: Posterior samples of the latent mean parameter.
        """
        # Closed-form posterior variance and mean
        sigma_n_squared = 1 / (self.observations_num / self.loss.var + 1 / self.prior.var)
        mu_n = sigma_n_squared * (self.observations_num * self.x_bar / self.loss.var + self.prior.mu / self.prior.var)
        self.mu_n = mu_n
        sigma_n = np.sqrt(sigma_n_squared)

        return np.random.normal(loc=mu_n, scale=sigma_n, size=n_samples).reshape(n_samples, -1)
        # return np.full((n_samples, 1), 3.0)

    def prior_score(self, x: ArrayLike) -> np.ndarray:
        """Score function of the prior.

        Args:
            x (ArrayLike): Points at which to evaluate the score.

        Returns:
            np.ndarray: Score values.
        """
        return self.prior.grad_log_pdf(x)

    def loss_score(self, x: ArrayLike) -> np.ndarray:
        """
        Score function of the likelihood, treating x as the latent variable (mean of Gaussian).
        Evaluates the gradient of log-likelihood for observed data.

        Args:
            x (ArrayLike): Latent variable values (e.g., mean) to evaluate the score at.
                           Should be shape (n_samples, 1)

        Returns:
            np.ndarray: Score values with shape (n_samples, 1)
        """
        return self.loss_lr * self.observations_num * (self.x_bar-x) / self.loss.var  # Shape (n_samples, 1)

    def posterior_score(self, x: ArrayLike) -> np.ndarray:
        """Score function of the posterior (prior + likelihood).

        Args:
            x (ArrayLike): Points at which to evaluate the score.

        Returns:
            np.ndarray: Score values.
        """
        ps = self.prior_score(x)
        ls = self.loss_score(x)
        return self.prior_score(x) + self.loss_score(x)
