import numpy as np

from src.bayesian_model.base import BayesianModel


class SimpleGaussianModel(BayesianModel):
    """
    Univariate Gaussian likelihood with Gaussian or LogNormal prior.
    """

    def __init__(self, data_config):
        super().__init__(data_config)

    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Sample from conjugate posterior: Normal(mu_n, sigma_n^2).
        """
        sigma_n2 = 1 / (self.observations_num / self.loss.var + 1 / self.prior.var)
        mu_n = sigma_n2 * (self.observations_num * self.x_bar / self.loss.var + self.prior.mu / self.prior.var)

        self.mu_n = mu_n
        sigma_n = np.sqrt(sigma_n2)

        return np.random.normal(mu_n, sigma_n, size=(n_samples, 1))


class MultivariateGaussianModel(BayesianModel):
    """
    Multivariate Gaussian likelihood with Gaussian prior on the mean.
    """

    def __init__(self, data_config):
        super().__init__(data_config)
        self.dim = self.observations.shape[1]

    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Closed-form posterior for mean with known covariance.
        """
        mu0, Sigma0 = self.prior.mu, self.prior.cov
        Sigma_obs = self.loss.cov

        Sigma_obs_inv = np.linalg.inv(Sigma_obs)
        Sigma0_inv = np.linalg.inv(Sigma0)

        Sigma_n_inv = self.observations_num * Sigma_obs_inv + Sigma0_inv
        Sigma_n = np.linalg.inv(Sigma_n_inv)

        mu_n = Sigma_n @ (self.observations_num * Sigma_obs_inv @ self.x_bar + Sigma0_inv @ mu0)

        self.mu_n = mu_n
        self.Sigma_n = Sigma_n

        return np.random.multivariate_normal(mu_n, Sigma_n, size=n_samples)
