import numpy as np
from scipy.stats import t
from .base import BaseDistribution

class StudentT(BaseDistribution):
    """
    Student's t-distribution.

    Parameters
    ----------
    df : float
        Degrees of freedom.
    loc : float
        Location (mean) parameter.
    scale : float
        Scale parameter.
    """

    def __init__(self, df: float, loc: float, scale: float):
        self.df = df
        self.loc = loc
        self.scale = scale

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return t.rvs(self.df, loc=self.loc, scale=self.scale, size=n_samples)

    def pdf(self, x: np.ndarray) -> float:
        return t.pdf(x, df=self.df, loc=self.loc, scale=self.scale)

    def log_pdf(self, x: np.ndarray) -> float:
        return t.logpdf(x, df=self.df, loc=self.loc, scale=self.scale)

    def grad_log_pdf(self, x: np.ndarray) -> np.ndarray:
        v = self.df
        z = (x - self.loc) / self.scale
        return -((v + 1) * z) / (v + z ** 2) / self.scale

    def grad_pdf(self, x: np.ndarray) -> np.ndarray:
        return self.grad_log_pdf(x) * self.pdf(x)