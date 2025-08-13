import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type
import numpy as np
import pandas as pd
from posteriordb import PosteriorDatabase

from src.utils.typing import ArrayLike


class KilpisjarviBayesianModel(ABC):
    def __init__(self, data_config: Any):
        """
        Bayesian model for Kilpisjarvi data (from posteriordb).
        """
        self.true_dgp = data_config.true_dgp
        self.loss_lr: float = data_config.loss_lr
        self.loss: Any = data_config.loss
        self.prior: Any = data_config.candidate_prior
        self.prior_init: Any = data_config.base_prior
        self.loss_lr_init: float = data_config.loss_lr
        self.pdb_path = data_config.pdb_path
        self.pdb_model_name = data_config.pdb_model_name
        self.warmup = getattr(data_config, "warmup", 0)

        # Prepare observations and initial posterior samples from posteriordb
        (self.observations,
         self.posterior_samples_init,
         self.posterior_sample_colnames) = self._prepare_data()
        self.observations_num = self.observations.shape[0]
        self.x_bar: np.ndarray = np.mean(self.observations, axis=0)

    def _prepare_observations(self, datavals: Dict[str, Any]) -> np.ndarray:
        """
        Convert posteriordb datavals to observation matrix (n, d).
        Expected keys: 'x', 'y', 'N'
        """
        x = np.asarray(datavals["x"])
        y = np.asarray(datavals["y"])
        self.loss.set_data(x, y)
        N = int(datavals["N"])
        if x.shape[0] != N or y.shape[0] != N:
            raise ValueError(f"Mismatch: N={N}, len(x)={x.shape[0]}, len(y)={y.shape[0]}")
        obs = np.column_stack([x, y])
        return obs.astype(float)

    def _norm_warmup(self, warmup, n_chains: int):
        """Return a list[int] of per-chain warmup counts."""
        if isinstance(warmup, (list, tuple)):
            if len(warmup) != n_chains:
                raise ValueError(f"warmup has length {len(warmup)} but there are {n_chains} chains.")
            return [max(0, int(w)) for w in warmup]
        return [max(0, int(warmup))] * n_chains

    def _concat_chains(self, series_with_chains: pd.Series,
                       warmup=0, max_draws=None) -> np.ndarray:
        """
        Concatenate arrays from all chains for a single parameter with warmup trimming.
        Each cell: array of draws for that chain of shape (draws, ...) .
        """
        raw = [np.asarray(a) for a in series_with_chains.to_numpy()]
        n_chains = len(raw)
        warmups = self._norm_warmup(warmup, n_chains)

        trimmed = []
        for i, arr in enumerate(raw):
            w = min(warmups[i], arr.shape[0])
            if warmups[i] > arr.shape[0]:
                warnings.warn(f"Warmup ({warmups[i]}) exceeds chain {i} draws ({arr.shape[0]}). Using {w}.")
            sl = arr[w:]
            if max_draws is not None:
                sl = sl[:max_draws]
            trimmed.append(sl)

        return np.concatenate(trimmed, axis=0)

    def _flatten_param_draws(self, name: str, arr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Flatten a parameter draw array (total_draws, *param_shape) to (total_draws, k)
        and return column names.
        """
        if arr.ndim == 1:  # (total_draws,)
            flat = arr.reshape(-1, 1)
            cols = [name]
        else:
            total_draws = arr.shape[0]
            param_shape = arr.shape[1:]
            k = int(np.prod(param_shape))
            flat = arr.reshape(total_draws, k)

            # Create readable column names for vector/tensor params
            if len(param_shape) == 1:
                cols = [f"{name}[{i+1}]" for i in range(param_shape[0])]
            else:
                # multi-index like name[i,j,k]
                from itertools import product
                indices = product(*[range(1, s+1) for s in param_shape])
                cols = [f"{name}[{','.join(map(str, idx))}]" for idx in indices]
        return flat, cols

    def _prepare_posterior_draws(
            self, df: pd.DataFrame, warmup=0, max_draws=None
    ) -> Tuple[np.ndarray, List[str]]:
        per_param_mats, per_param_cols = [], []

        for param in df.columns:
            concatenated = self._concat_chains(df[param], warmup=warmup, max_draws=max_draws)
            flat, cols = self._flatten_param_draws(param, concatenated)
            per_param_mats.append(flat)
            per_param_cols.extend(cols)

        samples = np.hstack(per_param_mats).astype(float)
        return samples, per_param_cols

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        my_pdb = PosteriorDatabase(self.pdb_path)
        posterior = my_pdb.posterior(self.pdb_model_name)
        code_stan = posterior.model.code("stan")

        datavals = posterior.data.values()
        observations = self._prepare_observations(datavals)

        reference_df = pd.DataFrame(posterior.reference_draws())
        posterior_samples, colnames = self._prepare_posterior_draws(
            reference_df,
            warmup=self.warmup,
        )
        return observations, posterior_samples, colnames

    def sample_from_base_prior(self, n_samples: int = 1000) -> np.ndarray:
        """
        Draw samples from the posterior distribution.
        """
        return self.prior_init.sample(n_samples)

    def set_prior_parameters(self, params: Dict[str, Any], distribution_cls: Type) -> None:
        """
        Set or update the prior distribution.

        Args:
            params: Dictionary of prior parameters.
            distribution_cls: Distribution class to instantiate the prior.
        """
        self.prior = distribution_cls(**params)


    def set_lr_parameter(self, lr: float) -> None:
        """
        Set or update the loss function parameters.
        """
        self.loss_lr = lr

    def prior_score(self, x: ArrayLike) -> np.ndarray:
        """Compute gradient of log prior."""
        return self.prior.grad_log_pdf(x)

    def loss_score(self, x: ArrayLike, multiply_by_lr: bool = True) -> np.ndarray:
        """Compute gradient of log likelihood (scaled by learning rate)."""
        grad = self.loss.grad_log_pdf(x, self.x_bar, self.observations_num)
        return self.loss_lr * grad if multiply_by_lr else grad

    def posterior_score(self, x: ArrayLike) -> np.ndarray:
        """Compute posterior score (prior + likelihood)."""
        prior_grad = self.prior_score(x)
        loss_grad = self.loss_score(x)
        return prior_grad + loss_grad

    def jacobian_sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """Return Jacobian of sufficient statistics."""
        return self.prior.grad_sufficient_statistics(x)

    def grad_log_base_measure(self, x: np.ndarray) -> np.ndarray:
        """Gradient of log base measure."""
        return self.prior.grad_log_base_measure(x)
