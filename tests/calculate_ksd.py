import numpy as np
import hydra
from hydra.utils import instantiate, get_original_cwd
import os

from src.discrepancies.posterior_ksd import PosteriorKSD
from src.plots.ksd_across_parameter_space import (plot_ksd_heatmaps, plot_ksd_line_plots, plot_ksd_multi_line_plots,
                                                  plot_ksd_multi_line_plots_with_minima, plot_ksd_multi_line_plots_with_error_bands,
                                                  plot_distribution_of_optimal_mu0)
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
    param_names = [param_name + "_0" for param_name in param_names]
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


@hydra.main(config_path="../configs/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_ksd_across_observations(cfg):
    obs_nums = [10, 100, 1000]
    mu_vals = np.round(np.linspace(-10, 10, 21), 2)
    fixed_sigma = 4.0
    repeats = 500  # number of repeated runs per setting

    ksd_results = {}  # will store list of KSDs for each (obs_num, mu_0)

    for obs_num in obs_nums:
        cfg.data.observations_num = obs_num
        model = instantiate(cfg.model, data_config=cfg.data)

        for mu_0 in mu_vals:
            ksd_list = []
            for _ in range(repeats):
                posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
                observations = np.array(model.observations).reshape(-1, 1)
                kernel = instantiate(cfg.ksd.kernel, reference_data=observations)

                model.set_prior_parameters({'mu': mu_0, 'sigma': fixed_sigma})
                ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
                ksd = ksd_estimator.estimate_ksd()
                ksd_list.append(ksd)

            ksd_results[(obs_num, mu_0)] = ksd_list
            median_ksd = np.median(ksd_list)
            std_ksd = np.std(ksd_list)
            print(f"obs_num={obs_num}, mu_0={mu_0}, mu_n={model.mu_n}, Median KSD={median_ksd:.4f}, ±3*STD={3 * std_ksd:.4f}")

    if cfg.flags.plots.generate_plots.line_plot:
        plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
        output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
        plot_cfg = load_plot_config(plot_config_path)

        # Prepare aggregated statistics for plotting
        median_results = {}
        lower_bound_results = {}
        upper_bound_results = {}

        for key, vals in ksd_results.items():
            median_results[key] = np.median(vals)
            std_val = np.std(vals)
            lower_bound_results[key] = median_results[key] - 3 * std_val
            upper_bound_results[key] = median_results[key] + 3 * std_val

        plot_ksd_multi_line_plots_with_error_bands(
            ksd_results=ksd_results,
            param_names=["obs.#", "mu_0"],
            plot_cfg=plot_cfg,
            output_dir=output_dir,
        )
        plot_distribution_of_optimal_mu0(
            ksd_results=ksd_results,
            param_names=["obs.#", "mu_0"],
            plot_cfg=plot_cfg,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    run()
    run_ksd_across_observations()