from itertools import product
from typing import Dict, List, Tuple, Any
import itertools
import numpy as np
from collections import OrderedDict
from scipy.spatial import ConvexHull
from itertools import product
from tqdm import tqdm

from src.distributions.gaussian import Gaussian, MultivariateGaussian
from src.discrepancies.posterior_ksd import PosteriorKSDParametric
from src.utils.distributions import DISTRIBUTION_MAP
from src.utils.files_operations import instantiate_from_target_str
from src.utils.distributions import is_basedistribution_like
from src.distributions.composite import CompositeProduct
from src.optimization.corners import get_corners


class OptimizationCornerPointsBase:
    def __init__(
        self,
        posterior_ksd: PosteriorKSDParametric,
        config: Dict,
        distribution_cls=Gaussian,
    ):
        """
        Base class to handle parametric quadratic form optimization.
        """
        self.posterior_ksd = posterior_ksd
        self.model = posterior_ksd.model
        self.distribution_cls = distribution_cls

        self.param_ranges: Dict[str, Tuple[float, float]] = config["parameters_box_range"]["ranges"]
        self.param_nums: Dict[str, int] = config["parameters_box_range"]["nums"]
        self.param_names = list(self.param_ranges.keys())

        self.Lambda_m_prior, self.b_m_prior, self.b_prior, self.b_cross_prior = self.posterior_ksd.compute_ksd_quadratic_form_for_prior()
        self.ksd_for_prior_init = self.posterior_ksd.compute_ksd_for_prior_term()
        self.Lambda_m_loss_lr, self.b_m_loss_lr, self.b_loss_lr, self.b_cross_loss_lr = self.posterior_ksd.compute_ksd_quadratic_form_for_loss()
        self.ksd_for_loss_init = self.posterior_ksd.compute_ksd_for_loss_term()
        self.ksd_for_cross_term = self.posterior_ksd.compute_cross_term()
        print(f"KSD terms: prior = {self.ksd_for_prior_init}, loss = {self.ksd_for_loss_init}, cross = {self.ksd_for_cross_term}")

    def _generate_corner_points(self) -> List[Dict[str, float]]:
        keys = list(self.param_ranges.keys())
        endpoints = [[bounds[0], bounds[1]] for bounds in self.param_ranges.values()]
        all_combinations = list(product(*endpoints))
        return [dict(zip(keys, values)) for values in all_combinations]

    def _evaluate_prior_qf_ksd(self, eta_tilde: np.ndarray) -> float:
        return eta_tilde @ self.Lambda_m_prior @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init


class OptimizationCornerPointsUnivariateGaussian(OptimizationCornerPointsBase):
    def __init__(
        self,
        posterior_ksd: PosteriorKSDParametric,
        config: Dict,
        distribution_cls=Gaussian,
    ):
        """
        Grid/corner quadratic form generation and optimization for univariate gaussian.
        """
        super().__init__(posterior_ksd=posterior_ksd, config=config, distribution_cls=distribution_cls)
        self.corner_points: List[Dict[str, float]] = self._generate_corner_points()
        self.parameter_grid = self._generate_full_parameter_grid()

    def _generate_full_parameter_grid(self) -> List[Tuple[float, ...]]:
        param_grids = [
            np.linspace(self.param_ranges[name][0], self.param_ranges[name][1], self.param_nums[name])
            for name in self.param_names
        ]
        return list(product(*param_grids))

    def evaluate_all_prior_corners(self) -> List:
        results = []

        for corner in self.corner_points:
            self.model.set_prior_parameters(corner, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            ksd_est = self._evaluate_prior_qf_ksd(eta_tilde)
            results.append((corner, eta_tilde, ksd_est))
            print(f"Corner: {corner} => Estimated KSD: {ksd_est:.3f}")

        results.sort(key=lambda x: x[2], reverse=True)

        return results

    def evaluate_all_lr_corners(self) -> List:
        results = []

        for corner in self.corner_points:
            self.model.set_lr_parameter(corner["lr"])
            lr = self.model.loss_lr
            ksd_est = lr**2 * self.Lambda_m_loss_lr + self.b_m_loss_lr * lr + self.ksd_for_prior_init
            results.append((corner, ksd_est))
            print(f"Corner: {corner} => Estimated KSD: {ksd_est:.6f}")

        return results

    def evaluate_all_prior_combinations(self) -> Tuple[List, List]:
        results = []

        for values in self.parameter_grid:
            param_dict = dict(zip(self.param_names, values))
            self.model.set_prior_parameters(param_dict, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            ksd_est = self._evaluate_prior_qf_ksd(eta_tilde)
            results.append((param_dict, eta_tilde, ksd_est))
            print(f"Corner: {param_dict} => Estimated KSD: {ksd_est:.6f}")

        return results, self.corner_points


class OptimizationCornerPointsMultivariateGaussian(OptimizationCornerPointsBase):
    def __init__(
        self,
        posterior_ksd: PosteriorKSDParametric,
        config: Dict,
        distribution_cls=MultivariateGaussian,
    ):
        """
        Grid/corner quadratic form generation and optimization for multivariate gaussian.
        """
        super().__init__(posterior_ksd=posterior_ksd, config=config, distribution_cls=distribution_cls)
        self.parameter_grid = self._generate_full_parameter_grid()
        self.distribution_corner_points: List[Dict[str, float]] = self._generate_corner_points()

    def _generate_mu_grid(self) -> List:
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
            cov = np.zeros((2, 2))
            for idx, k in enumerate(keys):
                i, j = map(int, k.split('_'))
                cov[i, j] = values[idx]
            cov_matrices.append(cov)
        return cov_matrices

    def _generate_full_parameter_grid(self) -> Dict:
        mu_grid = self._generate_mu_grid()
        cov_grid = self._generate_cov_grid()
        parameter_grid = {}

        for mu, cov in itertools.product(mu_grid, cov_grid):
            try:
                dist = self.distribution_cls(mu=np.array(mu), cov=np.array(cov))
            except Exception as e:
                print(f"Exception: {e} while initializing the distribution with mu={mu}, cov={cov}.")
                continue

            eta = dist.augmented_natural_parameters()
            cov_key = tuple(tuple(row) for row in cov)
            parameter_grid[(mu, cov_key)] = {
                "natural_parameters": eta,
                "distribution": dist
            }

        return parameter_grid

    def _generate_corner_points(self) -> List[Dict[str, float]]:
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

    def evaluate_all_prior_corners(self) -> List:
        results = []

        for corner_distribution in self.distribution_corner_points:
            params = corner_distribution.parameters_dict
            self.model.set_prior_parameters(params, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            ksd_est = self._evaluate_prior_qf_ksd(eta_tilde)
            results.append((params, eta_tilde, ksd_est))
            print(f"Corner: {params} => Estimated KSD: {ksd_est:.6f}")

        return results

    def evaluate_all_prior_combinations(self) -> Tuple:
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
        posterior_ksd: PosteriorKSDParametric,
        config: Dict,
        distribution_cls=Gaussian,
    ):
        """
        Grid/corner quadratic form generation and optimization for Inverse Wishart.
        """
        super().__init__(posterior_ksd=posterior_ksd, config=config, distribution_cls=distribution_cls)
        self.parameter_grid = self._generate_full_parameter_grid()
        self.distribution_corner_points: List[Dict[str, float]] = self._generate_corner_points()

    def _generate_df_grid(self) -> np.ndarray:
        return np.linspace(self.param_ranges["df"][0], self.param_ranges["df"][1], self.param_nums["df"])

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

    def _generate_full_parameter_grid(self) -> Dict:
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

    def _generate_corner_points(self) -> List:
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

    def evaluate_all_prior_corners(self) -> List:
        results = []

        for corner_distribution in self.distribution_corner_points:
            params = corner_distribution.parameters_dict
            self.model.set_prior_parameters(params, distribution_cls=self.distribution_cls)
            eta_tilde = self.model.prior.augmented_natural_parameters()
            ksd_est = eta_tilde @ self.Lambda_m_prior @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init
            results.append((params, eta_tilde, ksd_est))
            print(f"Corner: {params} => Estimated KSD: {ksd_est:.6f}")

        return results

    def evaluate_all_prior_combinations(self) -> Tuple:
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


class OptimizationCornerPointsCompositePrior:
    """
    Corner/grid quadratic-form optimization for a Composite prior.
    The model is expected to accept composite prior parameters via either:
        - model.set_composite_prior_parameters(params, combine_rule=...), or
        - model.set_prior_parameters({"components": params, "combine_rule": ...}, distribution_cls="CompositePrior")
    """

    def __init__(self, posterior_ksd: PosteriorKSDParametric, config: Dict, loss_config: Dict, precomputed_qfs: bool = False):
        self.posterior_ksd = posterior_ksd
        self.combine_rule: str = config.get("combine_rule", "product")
        self.components_cfg: List[Dict[str, Any]] = list(config["components"])
        self._validate_components()

        # QFs for loss
        self.loss_config = loss_config
        self.loss_lr_corners= self._generate_loss_lr_corner_points()
        self.loss_lr_full_grid = self._generate_full_loss_lr_grid()
        self.ksd_for_prior_init = self.posterior_ksd.compute_ksd_for_prior_term()
        self.Lambda_m_loss_lr, self.b_m_loss_lr, self.b_loss_lr, self.b_cross_loss_lr = self.posterior_ksd.compute_ksd_quadratic_form_for_loss()

        # Per-component corner lists and full grids
        self.component_corners_lambda: List[List[Dict[str, float]]] = [
            self._generate_component_corners(comp_cfg) for comp_cfg in self.components_cfg
        ]
        self.component_grids_lambda: List[List[Dict[str, float]]] = [
            self._generate_component_full_grid(comp_cfg) for comp_cfg in self.components_cfg
        ]
        # Map each component grid to eta and keep only eta-corners
        self.component_corner_records = [
            self._component_corners_from_predefined(comp_cfg)
            for comp_cfg in self.components_cfg
        ]
        # Expose lambda-params of eta-corners for your existing loops
        self.component_corners = [
            [rec["params"] for rec in recs] for recs in self.component_corner_records
        ]

        self._reset_prior_baseline_for_qf()

        # QFs for prior
        if not precomputed_qfs:
            (
                self.Lambda_m_prior,
                self.b_m_prior,
                self.b_prior,
                self.b_cross_prior,
            ) = self.posterior_ksd.compute_ksd_quadratic_form_for_prior()
            self.ksd_for_loss_init = self.posterior_ksd.compute_ksd_for_loss_term()
        else:
            (
                self.Lambda_m_prior,
                self.b_m_prior,
                self.b_prior,
                self.b_cross_prior,
            ) = self.posterior_ksd.precomputed_ksd_quadratic_form_for_prior
            self.ksd_for_loss_init = self.posterior_ksd.precomputed_ksd_for_loss_term


    def _component_corners_from_predefined(self, comp_cfg):
        """
        Prefer analytic corners when available, else fall back to grid + hull.
        Returns: [{"params": lambda_dict, "eta": eta_vec}, ...]
        """
        fam = comp_cfg["family"]
        ranges_cfg = comp_cfg["parameters_box_range"]["ranges"]
        ranges_tuples = {k: (float(v[0]), float(v[1])) for k, v in ranges_cfg.items()}
        recs = get_corners(fam, ranges_tuples)
        recs = [{"params": r["params"], "eta": np.asarray(r["eta"], dtype=float).ravel()} for r in recs]

        return recs

    def _component_gamma_corners_from_grid(self, comp_cfg, grid_lambda):
        """
        1) Map a dense grid in lambda-space to eta-space.
        2) Extract extreme points (hull vertices) in eta-space.
        3) Return records [{"params": lambda_dict, "eta": eta_vec}, ...].
        """
        etas, lams = self._component_gamma_from_grid(comp_cfg, grid_lambda)
        extreme_idx = self._gamma_extreme_indices_via_hull(etas)
        recs = [{"params": lams[i], "eta": etas[i]} for i in extreme_idx]
        return recs

    def _component_gamma_from_grid(self, comp_cfg, grid_lambda):
        """Map a component's lambda-grid to eta-vectors."""
        etas = []
        ls = []
        for lam in grid_lambda:
            dist = self._materialize_component({"family": comp_cfg["family"], "params": lam})
            eta = dist.natural_parameters()
            etas.append(eta.ravel())
            ls.append(lam)
        return np.asarray(etas, float), ls

    def _gamma_extreme_indices_via_hull(self, etas: np.ndarray, decimals: int = 10):
        E = np.asarray(etas, float)
        E_unique, keep_idx = np.unique(np.round(E, decimals), axis=0, return_index=True)

        n, p = E_unique.shape
        if n <= p + 1:
            return sorted(keep_idx.tolist())

        hull = ConvexHull(E_unique)
        verts_unique = hull.vertices

        return sorted(int(keep_idx[i]) for i in verts_unique)

    def _reset_prior_baseline_for_qf(self) -> None:
        """
        Reset the model prior once at initialization so that the subsequent
        QF precomputations use a consistent composite prior distribution.
        Strategy:
            - "center": center of each component's grid/box
            - "first_corner": use the first corner of each component
        Note: any choice here is fine for your QF since only distribution form matters.
        """
        per_component_params = [corners[0] for corners in self.component_corners_lambda]
        composite_payload = self._build_payload(per_component_params)
        self._set_composite_on_model(composite_payload)

    def _build_payload(self, per_component_params: List[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        return {
            comp_cfg["name"]: {"family": comp_cfg["family"], "params": params}
            for comp_cfg, params in zip(self.components_cfg, per_component_params)
        }

    def _validate_components(self):
        names = [c["name"] for c in self.components_cfg]
        if len(set(names)) != len(names):
            raise ValueError("Component names in Composite prior must be unique.")
        for comp in self.components_cfg:
            fam = comp.get("family")
            if fam not in DISTRIBUTION_MAP or DISTRIBUTION_MAP[fam] is None:
                raise ValueError(f"Unknown or unavailable distribution family: {fam}")

            pbr = comp.get("parameters_box_range", {})
            if "ranges" not in pbr or "nums" not in pbr:
                raise ValueError(f"parameters_box_range must contain 'ranges' and 'nums' for component {comp['name']}.")

    def _generate_component_corners(self, comp_cfg: Dict) -> List[Dict[str, float]]:
        ranges: Dict[str, Tuple[float, float]] = comp_cfg["parameters_box_range"]["ranges"]
        keys = list(ranges.keys())
        endpoints = [[ranges[k][0], ranges[k][1]] for k in keys]
        return [dict(zip(keys, vals)) for vals in product(*endpoints)]

    def _generate_component_full_grid(self, comp_cfg: Dict) -> List[Dict[str, float]]:
        ranges: Dict[str, Tuple[float, float]] = comp_cfg["parameters_box_range"]["ranges"]
        nums: Dict[str, int] = comp_cfg["parameters_box_range"]["nums"]
        axes = [np.linspace(ranges[k][0], ranges[k][1], nums[k]) for k in ranges.keys()]
        return [dict(zip(ranges.keys(), vals)) for vals in product(*axes)]

    def _materialize_component(self, spec: Any) -> Any:
        if is_basedistribution_like(spec):
            return spec
        if isinstance(spec, dict) and "family" in spec:
            fam = spec["family"]
            params = spec.get("params", {k: v for k, v in spec.items() if k != "family"})
            cls = DISTRIBUTION_MAP.get(fam)
            if cls is None:
                raise ValueError(f"Unknown family '{fam}'. Available: {list(DISTRIBUTION_MAP.keys())}")
            return cls(**params)
        if isinstance(spec, dict) and "_target_" in spec:
            kwargs = {k: v for k, v in spec.items() if k != "_target_"}
            return instantiate_from_target_str(spec["_target_"], kwargs)

        return spec

    def _set_composite_on_model(self, updated_components: Dict[str, Dict[str, Any]]):
        """
        updated_components: mapping name -> {"family": ..., "params": ...} or instance/_target_ spec.
        This merges with the existing composite prior on the model so unspecified components are preserved.
        """
        self.posterior_ksd.model.back_to_prior_candidate()
        full_map = OrderedDict()
        try:
            prior = getattr(self.posterior_ksd.model, "prior", None)
            if isinstance(prior, CompositeProduct):
                for name, comp in zip(prior.names, prior.components):
                    full_map[name] = comp
        except Exception:
            pass

        for name, spec in updated_components.items():
            full_map[name] = self._materialize_component(spec)

        if not full_map:
            full_map = OrderedDict((name, self._materialize_component(spec)))

        self.posterior_ksd.model.set_composite_prior_parameters(full_map, combine_rule=self.combine_rule)

    def _evaluate_prior_qf_ksd(self, eta_tilde: np.ndarray) -> float:
        return float(eta_tilde @ self.Lambda_m_prior @ eta_tilde + self.b_m_prior @ eta_tilde + self.ksd_for_loss_init)

    def evaluate_all_prior_corners(self) -> Tuple:
        results = []
        total = int(np.prod([len(c) for c in self.component_corners]))

        for combo in tqdm(product(*self.component_corners), total=total, desc="Evaluating corners"):
            composite_payload = {
                comp_cfg["name"]: {"family": comp_cfg["family"], "params": params}
                for comp_cfg, params in zip(self.components_cfg, combo)
            }
            self._set_composite_on_model(composite_payload)
            eta_tilde = self.posterior_ksd.model.prior.augmented_natural_parameters()
            ksd_est = self._evaluate_prior_qf_ksd(eta_tilde)
            results.append((composite_payload, eta_tilde, ksd_est))

        results.sort(key=lambda x: x[2], reverse=True)
        print(f"Corner with the largest sensitivity {results[0][2]}: {results[0][0]}.")

        return results, results[0][0]

    def evaluate_all_prior_combinations(self) -> Tuple[List, List]:
        results = []
        for combo in product(*self.component_grids_lambda):
            composite_payload = {
                comp_cfg["name"]: {"family": comp_cfg["family"], "params": params}
                for comp_cfg, params in zip(self.components_cfg, combo)
            }
            self._set_composite_on_model(composite_payload)
            eta_tilde = self.posterior_ksd.model.prior.augmented_natural_parameters()
            ksd_est = self._evaluate_prior_qf_ksd(eta_tilde)
            results.append((composite_payload, eta_tilde, ksd_est))
            print(f"Composite grid: {composite_payload} => Estimated KSD: {ksd_est:.2f}")
        corner_summaries = [
            {"component": comp_cfg["name"], "corners": corners}
            for comp_cfg, corners in zip(self.components_cfg, self.component_corners_lambda)
        ]
        return results, corner_summaries

    def _generate_loss_lr_corner_points(self) -> List[Dict[str, float]]:
        param_ranges = self.loss_config["parameters_box_range"]["ranges"]
        keys = list(param_ranges.keys())
        endpoints = [[bounds[0], bounds[1]] for bounds in param_ranges.values()]
        all_combinations = list(product(*endpoints))
        return [dict(zip(keys, values)) for values in all_combinations]


    def _generate_full_loss_lr_grid(self) -> dict:
        param_ranges = self.loss_config["parameters_box_range"]["ranges"]
        param_nums = self.loss_config["parameters_box_range"]["nums"]
        lr_grid = {"lr":np.linspace(param_ranges["lr"][0], param_ranges["lr"][1], param_nums["lr"])}

        return lr_grid

    def evaluate_all_lr_corners(self) -> List:
        self.posterior_ksd.model.back_to_prior_candidate()
        results = []
        for corner in self.loss_lr_corners:
            self.posterior_ksd.model.set_lr_parameter(corner["lr"])
            lr = self.posterior_ksd.model.loss_lr
            ksd_est = lr**2 * self.Lambda_m_loss_lr + self.b_m_loss_lr * lr + self.ksd_for_prior_init
            results.append((corner, ksd_est))
            print(f"Corner: {corner} => Estimated KSD: {ksd_est:.6f}")

        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Corner with the largest sensitivity {results[0][1]}: {results[0][0]}.")

        return results

    def evaluate_all_lr_grid(self) -> List:
        self.posterior_ksd.model.back_to_prior_candidate()
        results = []
        for lr in self.loss_lr_full_grid["lr"]:
            self.posterior_ksd.model.set_lr_parameter(lr)
            lr = self.posterior_ksd.model.loss_lr
            ksd_est = lr**2 * self.Lambda_m_loss_lr + self.b_m_loss_lr * lr + self.ksd_for_prior_init
            results.append((lr, ksd_est))
            print(f"Lr: {lr} => Estimated KSD: {ksd_est:.6f}")

        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Lr with the largest sensitivity {results[0][1]}: {results[0][0]}.")

        return results