import numpy as np
from abc import ABC, abstractmethod
from typing import Union

from src.utils.typing import ArrayLike


class BaseLoss(ABC):
    """
    Abstract base class for log-likelihood (loss) objects.
    """

    @abstractmethod
    def grad_log_pdf(
        self,
        x: ArrayLike,
        x_bar: Union[ArrayLike, float],
        observations_num: int,
    ) -> np.ndarray:

        raise NotImplementedError