import yaml
import numpy as np
from pathlib import Path
import argparse
from itertools import product
from typing import Dict, Any, Callable, List, Tuple
import hydra
from hydra.utils import instantiate

from src.kernels.base import BaseKernel
from src.bayesian_model.base import BayesianModel
from src.discrepancies.posterior_ksd import PosteriorKSD
from src.utils.instantiation import get_class_from_path, instantiate_distribution
from src.plots.ksd_across_parameter_space import plot_ksd_heatmaps


def get_kernel(kernel_config: Dict[str, Any]) -> BaseKernel:
    """Instantiate a kernel from configuration.

    Args:
        kernel_config (Dict[str, Any]): Configuration for the kernel.

    Returns:
        BaseKernel: Instantiated kernel object.
    """
    cls = get_class_from_path(kernel_config['type'])
    params = kernel_config.get('parameters', {})
    return cls(**params)


@hydra.main(config_path="../configs/ksd_calculation/toy/", config_name="univariate_gaussian")
def run(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    # Instantiate the Bayesian model with runtime overrides
    model = instantiate(cfg.model, data_config=cfg.data)

    # Sample from the posterior
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)

    # Instantiate the kernel
    observations = np.array(model.observations).reshape(-1, 1)
    kernel = instantiate(cfg.ksd.kernel, reference_data=observations)

    # Compute initial KSD
    ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    # Perform grid search over prior parameters if defined
    if cfg.ksd.optimize:
        ksd_results = {}
        box_range = cfg.ksd.optimize.prior.parameters_box_range
        param_names = list(box_range.keys())
        param_ranges = [np.linspace(*box_range[name], num=5) for name in param_names]

        for values in np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(param_names)):
            prior_params = dict(zip(param_names, values))
            model.set_prior_parameters(prior_params)

            ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
            ksd = ksd_estimator.estimate_ksd()
            ksd_results[tuple(values)] = ksd
            print(f"Prior: {prior_params}, KSD: {ksd:.4f}")

        # Plot if needed
        if cfg.flags.plots.generate_plots.heatmap:
            plot_ksd_heatmaps(ksd_results, param_names)


if __name__ == "__main__":
    run()