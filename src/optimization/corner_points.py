from itertools import product
from typing import Dict, List, Tuple, Any
import itertools
import numpy as np

from src.distributions.gaussian import Gaussian, MultivariateGaussian
from src.distributions.inverse_wishart import InverseWishart
from src.discrepancies.posterior_ksd import PosteriorKSD
from src.utils.checkers import is_symmetric, is_psd, is_symmetric_and_psd


class OptimizationCornerPointsBase:
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
        self.param_nums: Dict[str, Tuple[float, float]] = config["parameters_box_range"]["nums"]
        self.param_names = list(self.param_ranges.keys())


class OptimizationCornerPointsUnivariateGaussian(OptimizationCornerPointsBase):
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
        super().__init__(posterior_ksd=posterior_ksd, config=config, distribution_cls=distribution_cls)
        self.corner_points: List[Dict[str, float]] = self._generate_corner_points()
        self.parameter_grid = self._generate_full_parameter_grid()

        self.Lambda_m_prior, self.b_m_prior, self.b_prior, self.b_cross_prior, self.C, self.JT_aug_T_prior = self.posterior_ksd.compute_ksd_quadratic_form_for_prior()
        self.ksd_for_prior_init = self.posterior_ksd.compute_ksd_for_prior_term()

        self.Lambda_m_loss_lr, self.b_m_loss_lr, self.b_loss_lr, self.b_cross_loss_lr = self.posterior_ksd.compute_ksd_quadratic_form_for_loss()
        self.ksd_for_loss_init = self.posterior_ksd.compute_ksd_for_loss_term()

    def _generate_corner_points(self) -> List[Dict[str, float]]:
        """
        Generate all 2^d corner combinations from parameter box.
        """
        keys = list(self.param_ranges.keys())
        endpoints = [[bounds[0], bounds[1]] for bounds in self.param_ranges.values()]
        all_combinations = list(product(*endpoints))
        return [dict(zip(keys, values)) for values in all_combinations]

    def _generate_full_parameter_grid(self):
        """
        Generate full parameter grid from parameter ranges.
        """
        param_grids = [np.linspace(self.param_ranges[name][0], self.param_ranges[name][1], self.param_nums[name]) for name in self.param_names]
        all_combinations = list(product(*param_grids))

        return all_combinations

    def evaluate_all_prior_corners(self) -> List[Tuple[Dict[str, float], float]]:
        """
        For each corner point:
        - update prior parameters
        - recompute KSD quadratic form
        - evaluate KSD(η̃) = η̃ᵀ Λ η̃ + bᵀ η̃ + C

        Returns:
            List of (corner_param_dict, eta_tilde, estimated_ksd)
        """
        results = []

        for corner in self.corner_points:
            self.model.set_prior_parameters(corner, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            ksd_est = eta_tilde @ self.Lambda_m_prior @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init
            results.append((corner, eta_tilde, ksd_est))
            print(f"Corner: {corner} => Estimated KSD: {ksd_est:.6f}")

        return results


    def evaluate_all_lr_corners(self) -> List[Tuple[Dict[str, float], float]]:
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
            self.model.set_lr_parameter(corner["lr"])
            lr = self.model.loss_lr
            ksd_est = lr**2 * self.Lambda_m_loss_lr + self.b_m_loss_lr * lr + self.ksd_for_prior_init
            results.append((corner, ksd_est))
            print(f"Corner: {corner} => Estimated KSD: {ksd_est:.6f}")

        return results

    def evaluate_all_prior_combinations(self) -> List[Tuple[Dict[str, float], np.ndarray, float]]:
        """
        Evaluates the KSD at all parameter combinations (grid, not just corners).

        Returns:
            List of (param_dict, eta_tilde, estimated_ksd)
        """
        results = []

        for values in self.parameter_grid:
            param_dict = dict(zip(self.param_names, values))
            self.model.set_prior_parameters(param_dict, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            ksd_for_loss = self.posterior_ksd.compute_ksd_for_loss_term()
            ksd_est = eta_tilde @ self.Lambda_m_prior @ eta_tilde + self.b_m_prior @ eta_tilde + ksd_for_loss
            results.append((param_dict, eta_tilde, ksd_est))
            print(f"Corner: {param_dict} => Estimated KSD: {ksd_est:.6f}")

        return results, self.corner_points


class OptimizationCornerPointsMultivariateGaussian(OptimizationCornerPointsBase):
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
        super().__init__(posterior_ksd=posterior_ksd, config=config, distribution_cls=distribution_cls)
        self.parameter_grid = self._generate_full_parameter_grid()
        self.distribution_corner_points: List[Dict[str, float]] = self._generate_corner_points()

        self.Lambda_m_prior, self.b_m_prior, self.b_prior, self.b_cross_prior, self.C, self.JT_aug_T_prior = self.posterior_ksd.compute_ksd_quadratic_form_for_prior()
        self.ksd_for_prior_init = self.posterior_ksd.compute_ksd_for_prior_term()

        self.Lambda_m_loss_lr, self.b_m_loss_lr, self.b_loss_lr, self.b_cross_loss_lr = self.posterior_ksd.compute_ksd_quadratic_form_for_loss()
        self.ksd_for_loss_init = self.posterior_ksd.compute_ksd_for_loss_term()


    def _generate_mu_grid(self) -> List[np.ndarray]:
        mu_ranges = self.param_ranges['mu']
        mu_nums = self.param_nums['mu']
        mu_axes = [
            np.linspace(*mu_ranges[dim], mu_nums[dim])
            for dim in sorted(mu_ranges.keys(), key=int)
        ]
        return list(itertools.product(*mu_axes))

    def _generate_cov_grid(self) -> List[np.ndarray]:
        cov_ranges = self.param_ranges['cov']
        cov_nums = self.param_nums['cov']
        keys = sorted(cov_ranges.keys())
        axes = [
            np.linspace(*cov_ranges[k], cov_nums[k]) for k in keys
        ]
        cov_matrices = []
        for values in itertools.product(*axes):
            cov = np.zeros((2, 2))  # assumes 2D, generalize if needed
            for idx, k in enumerate(keys):
                i, j = map(int, k.split('_'))
                cov[i, j] = values[idx]
            cov_matrices.append(cov)
        return cov_matrices

    def _generate_full_parameter_grid(self) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]]]:
        mu_grid = self._generate_mu_grid()
        cov_grid = self._generate_cov_grid()

        natural_params = []
        distributions = []
        parameter_grid = {}

        for mu, cov in itertools.product(mu_grid, cov_grid):
            try:
                dist = self.distribution_cls(mu=np.array(mu), cov=np.array(cov))
            except Exception as e:
                print(f"Exception: {e} while initializing the distribution with mu={mu}, cov={cov}.")
                continue

            eta = dist.augmented_natural_parameters()
            natural_params.append(eta)
            distributions.append(dist)
            cov_key = tuple(tuple(row) for row in cov)
            parameter_grid[(mu, cov_key)] = {
                "natural_parameters": eta,
                "distribution": dist
            }

        return parameter_grid

    def _generate_corner_points(self) -> List[Dict[str, float]]:
        """
        Generate all 2^d corner combinations from parameter box.
        """
        all_eta = np.stack([v["natural_parameters"] for v in self.parameter_grid.values()])
        eta_min = all_eta.min(axis=0)
        eta_max = all_eta.max(axis=0)

        corners = list(itertools.product(*zip(eta_min, eta_max)))
        corner_set = {tuple(np.round(corner, 8)) for corner in corners}

        selected_distributions = [
            v["distribution"]
            for v in self.parameter_grid.values()
            if tuple(np.round(v["natural_parameters"], 8)) in corner_set
        ]

        return selected_distributions


    def evaluate_all_prior_corners(self) -> List[Tuple[Dict[str, float], float]]:
        """
        For each corner point:
        - update prior parameters
        - recompute KSD quadratic form
        - evaluate KSD(η̃) = η̃ᵀ Λ η̃ + bᵀ η̃ + C

        Returns:
            List of (corner_param_dict, eta_tilde, estimated_ksd)
        """
        results = []

        for corner_distribution in self.distribution_corner_points:
            params = corner_distribution.parameters_dict
            self.model.set_prior_parameters(params, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            ksd_est = eta_tilde @ self.Lambda_m_prior @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init
            results.append((params, eta_tilde, ksd_est))
            print(f"Corner: {params} => Estimated KSD: {ksd_est:.6f}")

        return results

    def evaluate_all_prior_combinations(self) -> List[Tuple[Dict[str, float], np.ndarray, float]]:
        """
        Evaluates the KSD at all parameter combinations (grid, not just corners).

        Returns:
            List of (param_dict, eta_tilde, estimated_ksd)
        """
        results = []

        for values in self.parameter_grid:
            param_dict = dict(zip(self.param_names, values))
            self.model.set_prior_parameters(param_dict, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            ksd_est = eta_tilde @ self.Lambda_m_prior @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init
            results.append((param_dict, eta_tilde, ksd_est))
            print(f"Corner: {param_dict} => Estimated KSD: {ksd_est:.6f}")

        results.sort(key=lambda x: x[2], reverse=True)
        return results, self.distribution_corner_points


class OptimizationCornerPointsInverseWishart(OptimizationCornerPointsBase):
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
        super().__init__(posterior_ksd=posterior_ksd, config=config, distribution_cls=distribution_cls)
        self.parameter_grid = self._generate_full_parameter_grid()
        self.distribution_corner_points: List[Dict[str, float]] = self._generate_corner_points()

        self.Lambda_m_prior, self.b_m_prior, self.b_prior, self.b_cross_prior, self.C, self.JT_aug_T_prior = self.posterior_ksd.compute_ksd_quadratic_form_for_prior()
        self.ksd_for_prior_init = self.posterior_ksd.compute_ksd_for_prior_term()

        self.Lambda_m_loss_lr, self.b_m_loss_lr, self.b_loss_lr, self.b_cross_loss_lr = self.posterior_ksd.compute_ksd_quadratic_form_for_loss()
        self.ksd_for_loss_init = self.posterior_ksd.compute_ksd_for_loss_term()

    def _generate_df_grid(self) -> List[np.ndarray]:
        df_grid = np.linspace(self.param_ranges["df"][0], self.param_ranges["df"][1], self.param_nums["df"])

        return df_grid

    def _generate_scale_grid(self) -> List[np.ndarray]:
        scale_ranges = self.param_ranges['scale']
        scale_nums = self.param_nums['scale']
        keys = sorted(scale_ranges.keys())
        axes = [
            np.linspace(*scale_ranges[k], scale_nums[k]) for k in keys
        ]
        scale_matrices = []
        for values in itertools.product(*axes):
            scale = np.zeros((2, 2))  # assumes 2D, generalize if needed
            for idx, k in enumerate(keys):
                i, j = map(int, k.split('_'))
                scale[i, j] = values[idx]
            scale_matrices.append(scale)
        return scale_matrices

    def _generate_full_parameter_grid(self) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]]]:
        df_grid = self._generate_df_grid()
        scale_grid = self._generate_scale_grid()
        parameter_grid = {}

        for df, scale in itertools.product(df_grid, scale_grid):
            try:
                dist = self.distribution_cls(df=df, scale=np.array(scale))
            except Exception as e:
                print(f"Exception: {e} while initializing the distribution with df={df}, scale={scale}.")
                continue

            eta = dist.augmented_natural_parameters()
            scale_key = tuple(tuple(row) for row in scale)
            parameter_grid[(df, scale_key)] = {
                "natural_parameters": eta,
                "distribution": dist
            }

        return parameter_grid

    def _generate_corner_points(self) -> List[Dict[str, float]]:
        """
        Generate all 2^d corner combinations from parameter box.
        """
        all_eta = np.stack([v["natural_parameters"] for v in self.parameter_grid.values()])
        eta_min = all_eta.min(axis=0)
        eta_max = all_eta.max(axis=0)

        corners = list(itertools.product(*zip(eta_min, eta_max)))
        corner_set = {tuple(np.round(corner, 8)) for corner in corners}

        selected_distributions = [
            v["distribution"]
            for v in self.parameter_grid.values()
            if tuple(np.round(v["natural_parameters"], 8)) in corner_set
        ]

        return selected_distributions

    def evaluate_all_prior_corners(self) -> List[Tuple[Dict[str, float], float]]:
        """
        For each corner point:
        - update prior parameters
        - recompute KSD quadratic form
        - evaluate KSD(η̃) = η̃ᵀ Λ η̃ + bᵀ η̃ + C

        Returns:
            List of (corner_param_dict, eta_tilde, estimated_ksd)
        """
        results = []

        for corner_distribution in self.distribution_corner_points:
            params = corner_distribution.parameters_dict
            self.model.set_prior_parameters(params, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            ksd_est = eta_tilde @ self.Lambda_m_prior @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init
            results.append((params, eta_tilde, ksd_est))
            print(f"Corner: {params} => Estimated KSD: {ksd_est:.6f}")

        return results

    def evaluate_all_prior_combinations(self) -> List[Tuple[Dict[str, float], np.ndarray, float]]:
        """
        Evaluates the KSD at all parameter combinations (grid, not just corners).

        Returns:
            List of (param_dict, eta_tilde, estimated_ksd)
        """
        results = []

        for values in self.parameter_grid:
            param_dict = dict(zip(self.param_names, values))
            self.model.set_prior_parameters(param_dict, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            ksd_est = eta_tilde @ self.Lambda_m_prior @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init
            results.append((param_dict, eta_tilde, ksd_est))
            print(f"Corner: {param_dict} => Estimated KSD: {ksd_est:.6f}")

        results.sort(key=lambda x: x[2], reverse=True)
        return results, self.distribution_corner_points