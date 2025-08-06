import numpy as np

from src.bayesian_model.base import BayesianModel
from src.utils.typing import ArrayLike
from src.distributions.inverse_wishart import InverseWishart


class InverseWishartModel(BayesianModel):
    """
    Covariance estimation with known mean, Inverse Wishart prior.
    """

    def __init__(self, data_config):
        super().__init__(data_config)
        self.mu = data_config.true_dgp.mu
        self.dim = self.mu.shape[0]

    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Posterior for covariance matrix under known-mean Gaussian model.
        """
        centered = self.observations - self.mu
        S = centered.T @ centered

        df_post = self.prior.df + self.observations_num
        scale_post = self.prior.scale + S

        posterior = InverseWishart(df=df_post, scale=scale_post)
        return posterior.sample(n_samples)

    def prior_score(self, x: ArrayLike) -> np.ndarray:
        return self.prior.grad_log_pdf(self.devectorize_samples(x))

    def loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        grad = self.loss.grad_log_pdf_wrt_cov(self.devectorize_samples(x), self.observations)
        return self.loss_lr * grad if multiply_by_lr else grad

    def jacobian_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """Return Jacobian of sufficient statistics."""
        return self.prior.grad_sufficient_statistics(self.devectorize_samples(x))

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        """Gradient of log base measure."""
        return self.prior.grad_log_base_measure(self.devectorize_samples(x))

    def vectorize_samples(self, samples: np.ndarray) -> np.ndarray:
        return samples.reshape(samples.shape[0], -1)

    def devectorize_samples(self, vectors: np.ndarray) -> np.ndarray:
        n_samples, d_squared = vectors.shape
        d = int(np.sqrt(d_squared))
        if d * d != d_squared:
            raise ValueError("Each vector must represent a square matrix.")
        return vectors.reshape(n_samples, d, d)

