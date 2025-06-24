import numpy as np
from scipy.stats import invwishart
from .base import BaseDistribution

class InverseWishart(BaseDistribution):
    """
    Inverse Wishart distribution.

    Parameters
    ----------
    df : float
        Degrees of freedom.
    scale : np.ndarray
        Positive definite scale matrix.
    """

    def __init__(self, df: float, scale: np.ndarray):
        self.df = df
        self.scale = scale

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return invwishart.rvs(df=self.df, scale=self.scale, size=n_samples)

    def pdf(self, x: np.ndarray) -> float:
        return invwishart.pdf(x, df=self.df, scale=self.scale)

    def log_pdf(self, x: np.ndarray) -> float:
        return invwishart.logpdf(x, df=self.df, scale=self.scale)

    def grad_log_pdf(self, x: np.ndarray) -> np.ndarray:
        # Analytical gradient is complex — placeholder
        raise NotImplementedError("Analytical grad_log_pdf for Inverse Wishart is not implemented.")

    def grad_pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("grad_pdf not implemented due to lack of analytical gradient.")
