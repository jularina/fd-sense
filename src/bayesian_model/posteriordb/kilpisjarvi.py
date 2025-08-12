from abc import ABC, abstractmethod
from typing import Any, Dict, Type
import numpy as np
import pandas as pd
from posteriordb import PosteriorDatabase


class KilpisjarviBayesianModel(ABC):
    def __init__(self, data_config: Any):
        """
        Bayesian model for Kilpisjarvi data.
        """
        self.true_dgp = data_config.true_dgp
        self.loss_lr: float = data_config.loss_lr
        self.loss: Any = data_config.loss
        self.prior: Any = data_config.candidate_prior
        self.prior_init: Any = data_config.base_prior
        self.loss_lr_init: float = data_config.loss_lr
        self.pdb_path = data_config.pdb_path
        self.pdb_model_name = data_config.pdb_model_name

        # Prepare observations
        self.observations, self.posterior_samples_init = self._prepare_data()
        self.observations_num = self.observations.shape[0]
        self.x_bar: np.ndarray = np.mean(self.observations, axis=0)


    def _prepare_data(self) -> np.ndarray:
        """
        Prepare observations form posteriordb.
        """
        my_pdb = PosteriorDatabase(self.pdb_path)
        posterior = my_pdb.posterior(self.pdb_model_name)
        datavals = posterior.data.values()
        posterior_reference_draws = posterior.reference_draws()
        posterior_reference_draws = pd.DataFrame(posterior_reference_draws)
