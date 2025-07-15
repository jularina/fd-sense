import numpy as np
import hydra
from hydra.utils import instantiate, get_original_cwd
import os

from src.discrepancies.posterior_ksd import PosteriorKSD
from src.plots.ksd_across_parameter_space import plot_ksd_heatmaps, plot_ksd_line_plots, plot_ksd_multi_line_plots
from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.utils.files_operations import load_plot_config


def box_range_ksd_recalculation(cfg, model: BayesianModel, posterior_samples: np.ndarray[float], kernel: BaseKernel):
    """
    Recalculates KSD along all the possible hyperparameters combination across the ranges

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model loaded by Hydra.
        posterior_samples (np.ndarray[float]): Posterior samples
        kernel (BaseKernel): Kernel
    """
    ksd_results = {}
    box_cfg = cfg.ksd.optimize.prior.parameters_box_range
    param_names = list(box_cfg.ranges.keys())
    param_ranges = [
        np.round(np.linspace(*box_cfg.ranges[name], num=box_cfg.nums[name]), 2)
        for name in param_names
    ]

    for values in np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(param_names)):
        prior_params = dict(zip(param_names, values))
        model.set_prior_parameters(prior_params)

        ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()
        ksd_results[tuple(values)] = ksd
        print(f"Prior: {prior_params}, KSD: {ksd:.4f}")

    # Plot if needed
    if cfg.flags.plots.generate_plots.heatmap:
        plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
        output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
        plot_cfg = load_plot_config(plot_config_path)
        plot_ksd_heatmaps(ksd_results, param_names, plot_cfg, output_dir)

    if cfg.flags.plots.generate_plots.line_plot:
        plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
        output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
        plot_cfg = load_plot_config(plot_config_path)
        plot_ksd_line_plots(ksd_results, param_names, plot_cfg, output_dir)
        plot_ksd_multi_line_plots(ksd_results, param_names, plot_cfg, output_dir)


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

    # Perform grid search over parameters box ranges if defined
    if cfg.ksd.optimize:
        box_range_ksd_recalculation(cfg, model, posterior_samples, kernel)



if __name__ == "__main__":
    run()