from abc import ABC, abstractmethod
import numpy as np

class BaseDistribution(ABC):
    """
    Abstract base class for probability distributions in the exponential family.
    """

    @abstractmethod
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Draw samples from the distribution."""
        pass

    @abstractmethod
    def pdf(self, x: np.ndarray) -> float:
        """Compute the probability density function at point x."""
        pass

    @abstractmethod
    def grad_pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of the PDF at point x."""
        pass

    @abstractmethod
    def log_pdf(self, x: np.ndarray) -> float:
        """Compute the log PDF at point x."""
        pass

    @abstractmethod
    def grad_log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of the log PDF at point x."""
        pass