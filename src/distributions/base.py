from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class BaseDistribution(ABC):
    """
    Abstract base class for exponential family distributions.
    """

    @abstractmethod
    def sample(self, n_samples: int = 1) -> NDArray:
        """Draw samples from the distribution."""
        pass

    @abstractmethod
    def pdf(self, x: NDArray) -> NDArray:
        """Evaluate the probability density function."""
        pass

    @abstractmethod
    def log_pdf(self, x: NDArray) -> NDArray:
        """Evaluate the log-density function."""
        pass

    @abstractmethod
    def grad_pdf(self, x: NDArray) -> NDArray:
        """Gradient of the density."""
        pass

    @abstractmethod
    def grad_log_pdf(self, x: NDArray) -> NDArray:
        """Gradient of the log-density."""
        pass

    def augmented_natural_parameters(self) -> NDArray:
        """Natural parameters plus constant for exponential form."""
        return np.append(self.natural_parameters(), 1.0)

    @abstractmethod
    def natural_parameters(self) -> NDArray:
        """Return natural parameters for the exponential form."""
        pass

    def grad_log_base_measure(self, x: NDArray) -> NDArray:
        """Gradient of log base measure (default: zero)."""
        return np.zeros_like(x)