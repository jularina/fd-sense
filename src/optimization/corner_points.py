from itertools import product
from typing import Dict, List, Tuple, Any

import numpy as np

from src.distributions.gaussian import Gaussian, MultivariateGaussian
from src.distributions.inverse_wishart import InverseWishart
from src.discrepancies.posterior_ksd import PosteriorKSD


class OptimizationCornerPoints:
    def __init__(
        self,
        posterior_ksd: PosteriorKSD,
        config: Dict,
        distribution_cls=Gaussian,
    ):
        """
        Args:
            posterior_ksd: PosteriorKSD object (already includes model).
            config: Hyperparameter config with:
                - parameters_box_range:
                    - ranges: Dict[str, [float, float]]
                    - nums:   Dict[str, int] (not used here)
            distribution_cls: Prior distribution class (e.g., Gaussian).
        """
        self.posterior_ksd = posterior_ksd
        self.model = posterior_ksd.model
        self.distribution_cls = distribution_cls

        # Extract hyperparameter box corners
        self.param_ranges: Dict[str, Tuple[float, float]] = config["parameters_box_range"]["ranges"]
        self.corner_points: List[Dict[str, float]] = self._generate_corner_points()

    def _generate_corner_points(self) -> List[Dict[str, float]]:
        """
        Generate all 2^d corner combinations from parameter box.
        """
        keys = list(self.param_ranges.keys())
        endpoints = [[bounds[0], bounds[1]] for bounds in self.param_ranges.values()]
        all_combinations = list(product(*endpoints))
        return [dict(zip(keys, values)) for values in all_combinations]


    def _prepare_prior_params(self, corner: Dict[str, float]) -> Dict[str, Any]:
        """
        Transforms corner dict into actual prior kwargs, including
        scale matrix construction from scalar value.
        """
        if self.distribution_cls.__name__ == "InverseWishart":
            df = corner["df"]
            scale_scalar = corner["scale_scalar"]
            d = self.model.dim
            scale = scale_scalar * np.eye(d)
            return {"df": df, "scale": scale}
        else:
            return corner  # e.g., for Gaussian


    def evaluate_all_corners(self) -> List[Tuple[Dict[str, float], float]]:
        """
        For each corner point:
        - update prior parameters
        - recompute KSD quadratic form
        - evaluate KSD(η̃) = η̃ᵀ Λ η̃ + bᵀ η̃ + C

        Returns:
            List of (corner_param_dict, estimated_ksd)
        """
        results = []

        for corner in self.corner_points:
            prior_params = self._prepare_prior_params(corner)
            self.model.set_prior_parameters(prior_params, distribution_cls=self.distribution_cls)
            # Natural parameter vector (augmented)
            eta_tilde = self.model.prior.augmented_natural_parameters()  # shape (p+1,)

            # Compute KSD quadratic form components
            Lambda_m, b_m, b_prior, b_cross, C, JT_aug_T = self.posterior_ksd.compute_ksd_quadratic_form_for_prior()
            ksd_for_loss = self.posterior_ksd.compute_ksd_for_loss_term()

            # Compute full estimated KSD
            ksd_est = eta_tilde @ Lambda_m @ eta_tilde + b_m @ eta_tilde + ksd_for_loss

            results.append((corner, ksd_est))
            print(f"Corner: {corner} => Estimated KSD: {ksd_est:.6f}")

        return results