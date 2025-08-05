import numpy as np
from typing import Dict, Type

from src.bayesian_model.base import BayesianModel
from src.distributions.gaussian import Gaussian
from src.distributions.log_normal import LogNormal
from src.utils.typing import ArrayLike


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
        self.loss = data_config.loss
        self.prior = data_config.base_prior
        self.m = data_config.posterior_samples_num

    def set_prior_parameters(self, params: Dict[str, float], distribution_cls: Type = Gaussian | LogNormal) -> None:
        """Update prior parameters for optimization.

        Args:
            params (Dict[str, float]): Dictionary of new prior parameters.
            distribution_cls (Type): The distribution class to use (default is Gaussian).
        """
        self.prior = distribution_cls(**params)

    def set_lr_parameter(self, params: dict) -> None:
        """
        Update the loss.
        Args:
            params (dict): {"df": ..., "scale": ...}
        """
        self.loss_lr = params["lr"]

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

    def loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        """
        Score function of the likelihood, treating x as the latent variable (mean of Gaussian).
        Evaluates the gradient of log-likelihood for observed data.

        Args:
            x (ArrayLike): Latent variable values (e.g., mean) to evaluate the score at.
                           Should be shape (n_samples, 1)

        Returns:
            np.ndarray: Score values with shape (n_samples, 1)
        """
        if multiply_by_lr:
            return self.loss_lr * self.loss.grad_log_pdf(x, self.x_bar, self.observations_num)  # Shape (n_samples, 1)
        else:
            return self.loss.grad_log_pdf(x, self.x_bar, self.observations_num)

    def posterior_score(self, x: ArrayLike) -> np.ndarray:
        """Score function of the posterior (prior + likelihood).

        Args:
            x (ArrayLike): Points at which to evaluate the score.

        Returns:
            np.ndarray: Score values.
        """
        prior_score = self.prior_score(x)
        loss_score = self.loss_score(x)
        posterior_score = prior_score + loss_score

        return posterior_score, prior_score, loss_score

    def jacobian_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        # Returns J_T(θ), shape (m, d, p)
        return self.prior.grad_sufficient_statistics(x)

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        # Returns ∇ log h(θ), shape (m, d)
        return self.prior.grad_log_base_measure(x)


class MultivariateGaussianModel(BayesianModel):
    """
    A Bayesian model for multivariate Gaussian likelihood with Gaussian prior.
    Assumes data ~ Normal(mu, Sigma_obs), and prior ~ Normal(mu0, Sigma0).
    """

    def __init__(self, data_config):
        # True data generator and sample observations
        self.true_dgp = data_config.true_dgp
        self.observations_num = data_config.observations_num
        self.observations = self.true_dgp.sample(self.observations_num)  # shape (n_samples, dim)

        self.dim = self.observations.shape[1]
        self.x_bar = np.mean(self.observations, axis=0)  # sample mean vector shape (dim,)

        self.loss_lr = data_config.loss_lr
        self.loss = data_config.loss
        self.prior = data_config.base_prior
        self.m = data_config.posterior_samples_num

    def set_prior_parameters(self, params: Dict[str, np.ndarray],
                             distribution_cls: Type = Gaussian | LogNormal) -> None:
        """
        Update prior parameters for optimization.

        Args:
            params (Dict[str, np.ndarray]): New prior parameters (mean vector and covariance matrix).
            distribution_cls (Type): Distribution class to use (default Gaussian).
        """
        self.prior = distribution_cls(**params)

    def set_lr_parameter(self, params: dict) -> None:
        """
        Update the loss.
        Args:
            params (dict): {"df": ..., "scale": ...}
        """
        self.loss_lr = params["lr"]

    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Sample from the closed-form conjugate posterior of a multivariate Gaussian model.

        Returns:
            np.ndarray: Posterior samples of the latent mean vector, shape (n_samples, dim).
        """
        # Prior parameters
        mu0 = self.prior.mu  # shape (dim,)
        Sigma0 = self.prior.cov  # shape (dim, dim)

        # Likelihood variance (observation noise covariance)
        Sigma_obs = self.loss.cov  # shape (dim, dim)
        Sigma_obs_inv = np.linalg.inv(Sigma_obs)
        Sigma0_inv = np.linalg.inv(Sigma0)
        Sigma_n_inv = self.observations_num * Sigma_obs_inv + Sigma0_inv
        Sigma_n = np.linalg.inv(Sigma_n_inv)

        # Posterior mean: mu_n = Sigma_n @ (n * inv(Sigma_obs) @ x_bar + inv(Sigma0) @ mu0)
        mu_n = Sigma_n @ (self.observations_num * Sigma_obs_inv @ self.x_bar + Sigma0_inv @ mu0)

        self.mu_n = mu_n
        self.Sigma_n = Sigma_n

        return np.random.multivariate_normal(mean=mu_n, cov=Sigma_n, size=n_samples)

    def prior_score(self, x: ArrayLike) -> np.ndarray:
        """
        Score function (gradient of log pdf) of the prior.

        Args:
            x (ArrayLike): Points at which to evaluate the score, shape (n_samples, dim).

        Returns:
            np.ndarray: Score values, shape (n_samples, dim).
        """
        return self.prior.grad_log_pdf(x)

    def loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        """
        Score function (gradient of log likelihood) treating x as the latent mean vector.

        Args:
            x (ArrayLike): Latent variables, shape (n_samples, dim).

        Returns:
            np.ndarray: Score values, shape (n_samples, dim).
        """
        if multiply_by_lr:
            return self.loss_lr * self.loss.grad_log_pdf(x, self.x_bar, self.observations_num)
        else:
            return self.loss.grad_log_pdf(x, self.x_bar, self.observations_num)

    def posterior_score(self, x: ArrayLike) -> np.ndarray:
        """
        Score function of the posterior (prior + likelihood).

        Args:
            x (ArrayLike): Points at which to evaluate the score, shape (n_samples, dim).

        Returns:
            np.ndarray: Score values, shape (n_samples, dim).
        """
        prior_score = self.prior_score(x)
        loss_score = self.loss_score(x)
        posterior_score = prior_score + loss_score

        return posterior_score, prior_score, loss_score

    def jacobian_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        # Returns J_T(θ), shape (m, d, p)
        return self.prior.grad_sufficient_statistics(x)

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        # Returns ∇ log h(θ), shape (m, d)
        return self.prior.grad_log_base_measure(x)