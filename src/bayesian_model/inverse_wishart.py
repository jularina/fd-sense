import numpy as np
from src.bayesian_model.base import BayesianModel
from src.utils.typing import ArrayLike
from src.distributions.inverse_wishart import InverseWishart


class InverseWishartModel(BayesianModel):
    """
    Bayesian model where mean is known, and the covariance matrix of the multivariate normal
    is unknown and inferred using an Inverse Wishart prior.
    """

    def __init__(self, data_config):
        self.true_dgp = data_config.true_dgp
        self.observations_num = data_config.observations_num
        self.observations = self.true_dgp.sample(self.observations_num)  # (n, d)

        self.mu = data_config.known_mean  # known fixed mean, shape (d,)
        self.dim = self.mu.shape[0]
        self.x_bar = self.observations.mean(axis=0)

        self.loss_lr = data_config.loss_lr
        self.loss = data_config.loss
        self.prior = data_config.base_prior  # instance of InverseWishart

    def set_prior_parameters(self, params: dict) -> None:
        """
        Update the inverse Wishart prior.
        Args:
            params (dict): {"df": ..., "scale": ...}
        """
        self.prior = InverseWishart(**params)

    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Conjugate posterior for unknown covariance (with known mean) is also Inverse Wishart.

        Returns:
            np.ndarray: Samples of covariance matrices, shape (n_samples, d, d)
        """
        n = self.observations.shape[0]
        X = self.observations

        # Center observations around known mu
        centered = X - self.mu
        S = centered.T @ centered

        # Posterior parameters
        df_post = self.prior.df + n
        scale_post = self.prior.scale + S

        # Sample from posterior Inverse Wishart
        posterior = InverseWishart(df=df_post, scale=scale_post)
        samples = posterior.sample(n_samples)  # shape (n_samples, d, d)

        return samples

    def prior_score(self, x: ArrayLike) -> np.ndarray:
        """
        Score of the Inverse Wishart prior w.r.t. Sigma
        """
        return self.prior.grad_log_pdf(x)

    def loss_score(self, x: ArrayLike) -> np.ndarray:
        """
        Score of the log-likelihood w.r.t. Sigma for multivariate normal with known mu:
        \nabla_\Sigma \log p(X | mu, Sigma)
        """
        n = self.observations.shape[0]
        X = self.observations
        centered = X - self.mu
        S = centered.T @ centered  # empirical scatter matrix
        x_inv = np.linalg.inv(x)

        grad = 0.5 * (x_inv @ S @ x_inv - n * x_inv)
        return grad

    def posterior_score(self, x: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Posterior score = prior_score + loss_score
        """
        prior_grad = self.prior_score(x)
        loss_grad = self.loss_score(x)
        total_grad = prior_grad + self.loss_lr * loss_grad
        return total_grad, prior_grad, loss_grad

    def jacobian_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        return self.prior.grad_sufficient_statistics(x)

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        return self.prior.grad_log_base_measure(x)
