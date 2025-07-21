from src.distributions.gaussian import Gaussian, MultivariateGaussian
from src.distributions.log_normal import LogNormal


DISTRIBUTION_MAP = {
    "Gaussian": Gaussian,
    "LogNormal": LogNormal,
    "MultivariateGaussian": MultivariateGaussian,
}