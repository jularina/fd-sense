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



class ECMOModel(BayesianModel):
    """
    ECMO model (Moreno 2000 / Kass-Greenhouse):

        y1 ~ Bin(n1, p1),   y2 ~ Bin(n2, p2)
        eta1 = gamma - delta/2
        eta2 = gamma + delta/2
        p_i = sigmoid(eta_i)

    Parameter vector:
        theta = [gamma, delta]
    """

    def __init__(self, data_config):
        super().__init__(data_config)

        # observations: (2,2) with rows [y1,n1], [y2,n2]
        self.y, self.n = self._prepare_observations(data_config)
        self.observations_num = 2

        self.posterior_samples_init = self._prepare_array_from_presaved_samples(
            getattr(data_config, "posterior_samples_path", None),
            name="posterior"
        )
        self.prior_samples_init = self._prepare_array_from_presaved_samples(
            getattr(data_config, "prior_samples_path", None),
            name="prior"
        )
        self.m = len(self.posterior_samples_init)
        self.m_prior = len(self.prior_samples_init)

    def _prepare_array_from_presaved_samples(self, path: Optional[str], name: str) -> Optional[np.ndarray]:
        """Same helper as in BioassayModel."""
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

    def _prepare_observations(self, data_config: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads ECMO observations saved as:
            observations.npy = [[y1, n1],
                                [y2, n2]]
        Returns:
            y: shape (2,1) and n: shape (2,1)
        """
        obs_path = getattr(data_config, "observations_path", None)
        if obs_path is None:
            raise ValueError("data_config must provide 'observations_path' for ECMOModel.")

        if not os.path.isabs(obs_path):
            obs_path = os.path.join(get_original_cwd(), obs_path)

        obs = load_numpy_array(obs_path)
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (2, 2):
            raise ValueError(f"Expected ECMO observations of shape (2,2) [[y1,n1],[y2,n2]], got {obs.shape}")

        y = obs[:, 0].reshape(-1, 1)   # (2,1)
        n = obs[:, 1].reshape(-1, 1)   # (2,1)
        return y, n

    def loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        """Compute gradient of log likelihood (scaled by learning rate)."""
        grad = self.loss.grad_log_pdf(theta=x, y=self.y, n=self.n)
        return self.loss_lr * grad if multiply_by_lr else grad

    def reference_loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        """Compute gradient of reference log loss."""
        grad = self.loss.grad_log_pdf(theta=x, y=self.y, n=self.n)
        return self.loss_lr_init * grad if multiply_by_lr else grad

    def set_composite_prior_parameters(
        self,
        components: Dict[str, Any],
        combine_rule: str = "product",
        reset_from_init: bool = True,
    ) -> None:
        """
        Same method as BioassayModel: update components of a CompositeProduct prior.
        Intended usage:
            components = {"gamma": {"loc": 0.0, "scale": 0.419}, "delta": {"loc": 0.0, "scale": 1.099}}
        or:
            components = {"gamma": {"family": "normal", "params": {"mean": 0.0, "std": 1.0}}}
        """
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