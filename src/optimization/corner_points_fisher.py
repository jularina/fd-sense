from typing import Dict, List, Tuple, Any
import itertools
import numpy as np
from collections import OrderedDict
from scipy.spatial import ConvexHull
from itertools import product
from tqdm import tqdm

from distributions.inverse_wishart import InverseWishart
from src.distributions.gaussian import Gaussian, MultivariateGaussian
from src.distributions.gamma import Gamma
from src.discrepancies.posterior_ksd import PosteriorKSDParametric
from src.utils.distributions import DISTRIBUTION_MAP
from src.utils.files_operations import instantiate_from_target_str
from src.utils.distributions import is_basedistribution_like
from src.distributions.composite import CompositeProduct
from src.optimization.corners import get_corners


class OptimizationCornerPointsBase:
    def __init__(
        self,
        posterior_estimator,
        prior_config: Dict,
        loss_config: Dict,
        distribution_cls=Gaussian,
    ):
        """
        Base class to handle parametric quadratic form optimization.
        """
        self.posterior_estimator = posterior_estimator
        self.model = posterior_estimator.model
        self.distribution_cls = distribution_cls

        self.param_ranges: Dict[str, Tuple[float, float]] = prior_config["parameters_box_range"]["ranges"]
        self.param_nums: Dict[str, int] = prior_config["parameters_box_range"]["nums"]
        self.param_names = list(self.param_ranges.keys())

        self.lr_ranges: Dict[str, Tuple[float, float]] = loss_config["parameters_box_range"]["ranges"]["lr"]
        self.lr_nums: Dict[str, int] = loss_config["parameters_box_range"]["nums"]["lr"]

        self.Lambda_prior, self.b_prior = self.posterior_estimator.compute_fisher_quadratic_form_for_prior()

        self.parameter_grid = self._generate_full_parameter_grid()
        self.distribution_corner_points: List[Dict[str, float]] = self._generate_corner_points()
        self.lr_grid = self._generate_full_lr_grid()
        self.lr_corners = self._generate_lr_corners()

    def _generate_corner_points(self):
        eta_list = []
        dists = []
        for v in self.parameter_grid.values():
            eta_list.append(np.asarray(v["natural_parameters"], dtype=float))
            dists.append(v["distribution"])

        all_eta = np.vstack(eta_list)
        hull = ConvexHull(all_eta)
        vertex_idxs = sorted(set(hull.vertices.tolist()))
        return [dists[i] for i in vertex_idxs]

    def _evaluate_prior_qf(self, eta_tilde: np.ndarray) -> float:
        return eta_tilde @ self.Lambda_prior @ eta_tilde + self.b_prior @ eta_tilde

    def _generate_full_parameter_grid(self) -> Dict:
        pass

    def _generate_full_lr_grid(self):
        lr_grid = np.linspace(self.lr_ranges[0], self.lr_ranges[-1], self.lr_nums).tolist()

        return lr_grid

    def _generate_lr_corners(self):
        return [self.lr_ranges[0], self.lr_ranges[-1]]

    def evaluate_all_prior_corners(self) -> Tuple:
        results = []

        for corner_distribution in self.distribution_corner_points:
            params = corner_distribution.parameters_dict
            self.model.set_prior_parameters(params, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            est = self._evaluate_prior_qf(eta_tilde)
            results.append((params, eta_tilde, est))

        results.sort(key=lambda x: x[2], reverse=True)
        self.model.back_to_prior_candidate()

        return results, results[0][0]

    def evaluate_all_prior_combinations(self) -> Tuple:
        results = []

        for values in self.parameter_grid:
            param_dict = dict(zip(self.param_names, values))
            self.model.set_prior_parameters(param_dict, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            est = eta_tilde @ self.Lambda_prior @ eta_tilde + self.b_prior @ eta_tilde
            results.append((param_dict, eta_tilde, est))
            print(f"Corner: {param_dict} => Estimated obj: {est:.6f}")

        results.sort(key=lambda x: x[2], reverse=True)
        self.model.back_to_prior_candidate()

        return results


    def evaluate_all_lr_corners(self) -> List:
        results = []

        for lr_corner in self.lr_corners:
            self.model.set_lr_parameter(lr_corner)
            lr = self.model.loss_lr
            ksd_est = lr**2 * self.Lambda_loss_lr + self.b_loss_lr * lr
            results.append((lr_corner, ksd_est))
            print(f"Corner: {lr_corner} => Estimated obj: {ksd_est:.6f}")

        results.sort(key=lambda x: x[1], reverse=True)
        self.model.back_to_lr_init()

        return results


    def evaluate_full_lr_grid(self) -> List:
        results = []

        for lr in self.lr_grid:
            self.model.set_lr_parameter(lr)
            lr = self.model.loss_lr
            ksd_est = lr**2 * self.Lambda_loss_lr + self.b_loss_lr * lr
            results.append((lr, ksd_est))
            print(f"Corner: {lr} => Estimated obj: {ksd_est:.6f}")

        results.sort(key=lambda x: x[1], reverse=True)
        self.model.back_to_lr_init()

        return results


class OptimizationCornerPointsUnivariateGaussian(OptimizationCornerPointsBase):
    def __init__(
        self,
        posterior_estimator,
        prior_config: Dict,
        loss_config: Dict,
        distribution_cls=Gaussian,
    ):
        """
        Grid/corner quadratic form generation and optimization for univariate gaussian.
        """
        super().__init__(posterior_estimator=posterior_estimator, prior_config=prior_config, loss_config=loss_config, distribution_cls=distribution_cls)

    def _generate_mu_grid(self) -> List:
        mu_ranges = self.param_ranges['mu']
        mu_num = self.param_nums['mu']
        mu_grid = np.linspace(mu_ranges[0], mu_ranges[-1], mu_num).tolist()

        return mu_grid

    def _generate_sigma_grid(self) -> List[np.ndarray]:
        sigma_ranges = self.param_ranges['sigma']
        sigma_num = self.param_nums['sigma']
        sigma_grid = np.linspace(sigma_ranges[0], sigma_ranges[-1], sigma_num).tolist()

        return sigma_grid

    def _generate_full_parameter_grid(self) -> Dict:
        mu_grid = self._generate_mu_grid()
        sigma_grid = self._generate_sigma_grid()
        parameter_grid = {}

        for mu, sigma in itertools.product(mu_grid, sigma_grid):
            try:
                dist = self.distribution_cls(mu=mu, sigma=sigma)
            except Exception as e:
                print(f"Exception: {e} while initializing the distribution with mu={mu}, aigma={sigma}.")
                continue

            augmented_eta = dist.augmented_natural_parameters()
            eta = dist.natural_parameters()
            parameter_grid[(mu, sigma)] = {
                "augmented_natural_parameters": augmented_eta,
                "natural_parameters": eta,
                "distribution": dist
            }

        return parameter_grid