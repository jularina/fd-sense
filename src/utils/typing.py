import numpy as np
from typing import Union
from src.distributions.gaussian import Gaussian
from src.distributions.log_normal import LogNormal

ArrayLike = Union[np.ndarray, list]


DISTRIBUTION_MAP = {
    "Gaussian": Gaussian,
    "LogNormal": LogNormal,
}