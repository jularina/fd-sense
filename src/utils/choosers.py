from omegaconf import DictConfig

from src.optimization.corner_points import (
    OptimizationCornerPointsUnivariateGaussian,
    OptimizationCornerPointsInverseWishart,
    OptimizationCornerPointsMultivariateGaussian,
)
from src.discrepancies.posterior_ksd import PosteriorKSDParametric


def pick_optimizer(cfg: DictConfig, ksd_estimator: PosteriorKSDParametric):
    """Pick the corner-points optimizer from cfg.ksd.optimize.prior.*"""
    prior_cfg = cfg.ksd.optimize.prior
    if hasattr(prior_cfg, "Gaussian"):
        return OptimizationCornerPointsUnivariateGaussian(ksd_estimator, prior_cfg.Gaussian)
    if hasattr(prior_cfg, "MultivariateGaussian"):
        return OptimizationCornerPointsMultivariateGaussian(ksd_estimator, prior_cfg.MultivariateGaussian)
    if hasattr(prior_cfg, "InverseWishart"):
        return OptimizationCornerPointsInverseWishart(ksd_estimator, prior_cfg.InverseWishart)
    raise ValueError("No supported prior found under cfg.ksd.optimize.prior. Supported: Gaussian, MultivariateGaussian, InverseWishart")