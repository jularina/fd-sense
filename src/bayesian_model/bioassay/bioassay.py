import os
from hydra.utils import get_original_cwd
from typing import Optional
import copy
from typing import Any, Dict
import numpy as np

from src.utils.typing import ArrayLike
from src.bayesian_model.base import BayesianModel
from src.utils.files_operations import load_numpy_array
from src.distributions.composite import CompositeProduct
from src.utils.distributions import DISTRIBUTION_MAP
from src.utils.files_operations import instantiate_from_target_str
from src.utils.distributions import is_basedistribution_like


class BioassayModel(BayesianModel):
    """
    Bioassay Model
    """

    def __init__(self, data_config):
        super().__init__(data_config)
        self.x, self.y, self.n = self._prepare_observations(data_config)
        self.observations_num = self.x.shape[0]
        self.posterior_samples_init = self._prepare_array_from_presaved_samples(
            getattr(data_config, "posterior_samples_path", None),
            name="posterior"
        )
        if self.m is None:
            self.m = len(self.posterior_samples_init)

    def _prepare_array_from_presaved_samples(self, path: Optional[str], name: str) -> Optional[np.ndarray]:
        """
        Generic helper to load an array from a given path in config.
        Returns None if no path is given.
        """
        if path is None:
            return None
        if not os.path.isabs(path):
            path = os.path.join(get_original_cwd(), path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name.capitalize()} samples file not found: {path}")

        arr = load_numpy_array(path)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _prepare_observations(self, data_config: Any) -> tuple:
        obs_path = getattr(data_config, "observations_path", None)

        if not os.path.isabs(obs_path):
            obs_path = os.path.join(get_original_cwd(), obs_path)

        obs = load_numpy_array(obs_path)

        return obs[:,0].reshape(-1,1), obs[:,1].reshape(-1,1), obs[:,2].reshape(-1,1)

    def loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        """Compute gradient of log likelihood (scaled by learning rate)."""
        grad = self.loss.grad_log_pdf(theta=x, y=self.y, x=self.x, n=self.n)
        return self.loss_lr * grad if multiply_by_lr else grad

    def set_composite_prior_parameters(self, components: Dict[str, Any], combine_rule: str = "product",
                                       reset_from_init: bool = True, ) -> None:
        if combine_rule != "product":
            raise NotImplementedError("Only product composites are supported right now.")

        base = self.prior_init if reset_from_init else self.prior

        if not isinstance(base, CompositeProduct):
            raise TypeError("Expected CompositeProduct as base prior.")

        base_names = list(base.names)
        base_map = {n: copy.deepcopy(c) for n, c in zip(base.names, base.components)}
        new_map: Dict[str, Any] = dict(base_map)

        for name, spec in components.items():
            if is_basedistribution_like(spec):
                new_map[name] = spec
                continue

            if isinstance(spec, dict) and "_target_" in spec:
                kwargs = {k: v for k, v in spec.items() if k != "_target_"}
                new_map[name] = instantiate_from_target_str(spec["_target_"], kwargs)
                continue

            if isinstance(spec, dict) and "family" in spec:
                fam = spec["family"]
                params = spec.get("params", {k: v for k, v in spec.items() if k != "family"})
                cls = DISTRIBUTION_MAP.get(fam)
                if cls is None:
                    raise ValueError(f"Unknown family '{fam}'. Available: {list(DISTRIBUTION_MAP.keys())}")
                new_map[name] = cls(**params)
                continue

            if isinstance(spec, dict) and name in base_map:
                cls = base_map[name].__class__
                new_map[name] = cls(**spec)
                continue

            raise ValueError(
                f"Component '{name}' must be a BaseDistribution instance, "
                f"a dict with '_target_', a dict with 'family'/params, or bare params matching a base component."
            )

        ordered_map: Dict[str, Any] = {n: new_map[n] for n in base_names if n in new_map}
        for n, v in new_map.items():
            if n not in ordered_map:
                ordered_map[n] = v

        self.prior = CompositeProduct(distributions=ordered_map)