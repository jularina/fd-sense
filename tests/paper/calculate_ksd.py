from statistics import median

import numpy as np
import hydra
from hydra.utils import instantiate, get_original_cwd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy

from src.discrepancies.posterior_ksd import PosteriorKSD
from src.plots.paper.paper_funcs import *
from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.distributions.gaussian import MultivariateGaussian
from src.distributions.inverse_wishart import InverseWishart
from src.utils.files_operations import load_plot_config
from src.utils.distributions import DISTRIBUTION_MAP
from src.optimization.corner_points import OptimizationCornerPointsUnivariateGaussian, OptimizationCornerPointsInverseWishart, OptimizationCornerPointsMultivariateGaussian


def plots_across_gaussian_prior_parameters_ranges(cfg, model: BayesianModel, posterior_samples: np.ndarray[float], kernel: BaseKernel):
    """
    Recalculates KSD along all the possible hyperparameters combination across the ranges

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model loaded by Hydra.
        posterior_samples (np.ndarray[float]): Posterior samples
        kernel (BaseKernel): Kernel
    """
    ksd_results = {}
    box_cfg = cfg.ksd.optimize.prior.Gaussian.parameters_box_range
    distribution_cls = DISTRIBUTION_MAP["Gaussian"]
    param_names = list(box_cfg.ranges.keys())
    param_ranges = [
        np.round(np.linspace(*box_cfg.ranges[name], num=box_cfg.nums[name]), 2)
        for name in param_names
    ]
    for values in np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(param_names)):
        prior_params = dict(zip(param_names, values))
        model.set_prior_parameters(prior_params, distribution_cls=distribution_cls)

        ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()
        ksd_results[tuple(values)] = ksd
        print(f"Prior: {prior_params}, mu_n: {model.mu_n}, KSD: {ksd:.4f}")

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


def plots_across_gaussian_parameters_ranges_etas_quadratic_form(cfg, eta_ksd_results, corner_points):
    """
    Recalculates KSD along all the possible hyperparameters combination across the ranges

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model loaded by Hydra.
        posterior_samples (np.ndarray[float]): Posterior samples
        kernel (BaseKernel): Kernel
    """
    if cfg.flags.plots.generate_plots.line_plot:
        plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
        output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
        plot_cfg = load_plot_config(plot_config_path)
        plot_ksd_eta_surface(eta_ksd_results, corner_points, plot_cfg, output_dir)


def plots_across_gaussian_loss_lr_parameters_ranges(cfg, model: BayesianModel, posterior_samples: np.ndarray[float], kernel: BaseKernel):
    """
    Recalculates KSD along all the possible hyperparameters combination across the ranges

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model loaded by Hydra.
        posterior_samples (np.ndarray[float]): Posterior samples
        kernel (BaseKernel): Kernel
    """
    ksd_results = {}
    box_cfg = cfg.ksd.optimize.loss.GaussianLogLikelihood.parameters_box_range
    param_names = list(box_cfg.ranges.keys())
    param_ranges = [
        np.round(np.linspace(*box_cfg.ranges[name], num=box_cfg.nums[name]), 2)
        for name in param_names
    ]
    for values in np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(param_names)):
        params = dict(zip(param_names, values))
        model.set_lr_parameter(params)

        ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()
        ksd_results[values[0]] = ksd
        print(f"Lr: {params}, KSD: {ksd:.4f}")

    if cfg.flags.plots.generate_plots.line_plot:
        plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
        output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
        plot_cfg = load_plot_config(plot_config_path)
        plot_ksd_single_param(ksd_results, param_names[0], plot_cfg, output_dir)


def density_plot_across_gaussian_prior_parameter_set(cfg, model: BayesianModel, posterior_samples: np.ndarray[float], kernel: BaseKernel):
    """
    Recalculates KSD along all the possible hyperparameters combination across the ranges

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model loaded by Hydra.
        posterior_samples (np.ndarray[float]): Posterior samples
        kernel (BaseKernel): Kernel
    """
    ksd_results = {}
    box_cfg = cfg.ksd.optimize.prior.Gaussian.parameters_box_range
    param_names = list(box_cfg.ranges.keys())
    params = np.array([[-7, 1], [5, 3], [9, 2]])
    distribution_cls = DISTRIBUTION_MAP["Gaussian"]
    for values in params:
        prior_params = dict(zip(param_names, values))
        model.set_prior_parameters(prior_params, distribution_cls=distribution_cls)

        ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()
        ksd_results[tuple(values)] = ksd
        print(f"Prior: {prior_params}, mu_n: {model.mu_n}, KSD: {ksd:.4f}")

    # Plot if needed
    param_names = [param_name + "_0" for param_name in param_names]
    if cfg.flags.plots.generate_plots.density_plot:
        plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
        output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
        plot_cfg = load_plot_config(plot_config_path)
        plot_gaussian_prior_densities_by_ksd(ksd_results, param_names, cfg, plot_cfg, output_dir)


def density_plot_across_prior_parameter_sets(cfg, model: BayesianModel, posterior_samples: np.ndarray, kernel: BaseKernel):
    """
    Recalculates KSD across all prior hyperparameter combinations for each distribution.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model.
        posterior_samples (np.ndarray): Posterior samples.
        kernel (BaseKernel): Kernel.
    """
    all_ksd_results = {}
    param_values_dict = {"Gaussian":np.array([[-7, 1], [5, 3], [9, 2]]), "LogNormal":np.array([[1, 0.5]])}
    for dist_name, dist_cfg in cfg.ksd.optimize.prior.items():
        distribution_cls = DISTRIBUTION_MAP[dist_name]
        box_cfg = dist_cfg.parameters_box_range
        param_names = list(box_cfg.ranges.keys())
        param_values = param_values_dict[dist_name]
        dist_ksd_results = {}

        for values in param_values:
            prior_params = dict(zip(param_names, values))
            model.set_prior_parameters(prior_params, distribution_cls=distribution_cls)

            ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
            ksd = ksd_estimator.estimate_ksd()
            dist_ksd_results[tuple(values)] = ksd
            print(f"Dist: {dist_name}, Prior: {prior_params}, mu_n: {model.mu_n}, KSD: {ksd:.4f}")

        all_ksd_results[dist_name] = {
            "ksd": dist_ksd_results,
            "param_names": [p + "_0" for p in param_names],
            "distribution_cls": distribution_cls
        }

    # Plot all distributions
    if cfg.flags.plots.generate_plots.density_plot:
        plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
        output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
        plot_cfg = load_plot_config(plot_config_path)

        plot_prior_densities_by_ksd(
            all_ksd_data=all_ksd_results,
            cfg=cfg,
            plot_cfg=plot_cfg,
            output_dir=output_dir,
        )

def density_plot_across_multivariate_prior_parameter_sets(
    cfg,
    model,
    posterior_samples,
    kernel,
    qf_across_priors,
):
    # Define priors to test
    param_values_dict = {
        "MultivariateGaussian": [
            {"mu": np.array([0.0, 0.0]), "cov": np.eye(2)},
            {"mu": np.array([2.0, 3.0]), "cov": 0.5 * np.eye(2)},
            {"mu": np.array([-2.0, 1.0]), "cov": np.array([[1.0, 0.5], [0.5, 1.5]])},
        ]
    }

    all_ksd_results = {}
    dist_name = "MultivariateGaussian"
    distribution_cls = MultivariateGaussian
    param_values = param_values_dict[dist_name]
    dist_ksd_results = {}
    all_dists = {}

    for param_dict in param_values:
        model.set_prior_parameters(param_dict, distribution_cls=distribution_cls)
        ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()

        key = (tuple(param_dict["mu"].flatten()), tuple(param_dict["cov"].flatten()))
        dist_ksd_results[key] = ksd
        all_dists[key] = param_dict

        print(f"[INFO] Prior: {param_dict}, KSD: {ksd:.4f}")

    if cfg.flags.plots.generate_plots.density_plot:
        plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
        output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
        plot_cfg = load_plot_config(plot_config_path)
        plot_multivariate_priors_densities_by_ksd(
            all_params=all_dists,
            all_ksds=dist_ksd_results,
            output_dir=output_dir,
            plot_cfg=plot_cfg,
            true_theta=cfg.data.base_prior.mu,
            true_cov = cfg.data.base_prior.cov
        )
        plot_multivariate_joint_prior_densities_by_ksd(
            results=qf_across_priors,
            output_dir=output_dir,
            plot_cfg=plot_cfg,
            true_theta=cfg.data.base_prior.mu,
            true_cov = cfg.data.base_prior.cov
        )


def compute_ksd_for_setting(obs_num, mu_0, cfg_serialized, repeats, fixed_sigma):
    cfg = copy.deepcopy(cfg_serialized)  # each process gets its own copy
    cfg.data.observations_num = obs_num
    distribution_cls = DISTRIBUTION_MAP["Gaussian"]
    model = instantiate(cfg.model, data_config=cfg.data)

    ksd_list = []
    mu_ns = []
    for _ in range(repeats):
        posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
        mu_ns.append(model.mu_n)
        kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
        model.set_prior_parameters({'mu': mu_0, 'sigma': fixed_sigma}, distribution_cls=distribution_cls)
        ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()
        ksd_list.append(ksd)

    median_ksd = np.median(ksd_list)
    std_ksd = np.std(ksd_list)
    print(f"[PID {os.getpid()}] obs_num={obs_num}, mu_0={mu_0}, mu_n={median(mu_ns):.2f}, Median KSD={median_ksd:.4f}, ±3*STD={3 * std_ksd:.4f}")

    return (obs_num, mu_0), ksd_list


@hydra.main(config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian", version_base="1.3")
def run_ksd_across_various_observations_nums_parallel(cfg):
    obs_nums = [50, 100, 200]
    mu_vals = np.round(np.linspace(-10, 10, 21), 2)
    fixed_sigma = 2.0
    repeats = 100  # number of repeated runs per setting

    ksd_results = {}  # will store list of KSDs for each (obs_num, mu_0)

    tasks = [
        (obs_num, mu_0, copy.deepcopy(cfg), repeats, fixed_sigma)
        for obs_num in obs_nums
        for mu_0 in mu_vals
    ]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_ksd_for_setting, *args) for args in tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            key, ksd_list = future.result()
            ksd_results[key] = ksd_list

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
            param_names=["obs. num.", "mu_0"],
            plot_cfg=plot_cfg,
            output_dir=output_dir,
        )
        plot_distribution_of_optimal_mu0(
            ksd_results=ksd_results,
            param_names=["obs. num.", "mu_0"],
            plot_cfg=plot_cfg,
            output_dir=output_dir,
        )

@hydra.main(config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian", version_base="1.3")
def run_ksd_across_various_observations_nums(cfg):
    obs_nums = [50, 100, 200]
    mu_vals = np.round(np.linspace(-10, 10, 21), 2)
    fixed_sigma = 2.0
    repeats = 100  # number of repeated runs per setting
    distribution_cls = DISTRIBUTION_MAP["Gaussian"]

    ksd_results = {}  # will store list of KSDs for each (obs_num, mu_0)

    for obs_num in obs_nums:
        cfg.data.observations_num = obs_num
        model = instantiate(cfg.model, data_config=cfg.data)

        for mu_0 in mu_vals:
            ksd_list = []
            mu_ns = []
            for repeat in range(repeats):
                print(f"obs_num={obs_num}, mu_0={mu_0}, repeat={repeat}")
                posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
                mu_ns.append(model.mu_n)
                kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

                model.set_prior_parameters({'mu': mu_0, 'sigma': fixed_sigma}, distribution_cls=distribution_cls)
                ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
                ksd = ksd_estimator.estimate_ksd()
                ksd_list.append(ksd)

            ksd_results[(obs_num, mu_0)] = ksd_list
            median_ksd = np.median(ksd_list)
            std_ksd = np.std(ksd_list)
            print(f"obs_num={obs_num}, mu_0={mu_0}, mu_n={median(mu_ns)}, Median KSD={median_ksd:.4f}, ±3*STD={3 * std_ksd:.4f}")

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
            param_names=["obs. num.", "mu_0"],
            plot_cfg=plot_cfg,
            output_dir=output_dir,
        )
        plot_distribution_of_optimal_mu0(
            ksd_results=ksd_results,
            param_names=["obs. num.", "mu_0"],
            plot_cfg=plot_cfg,
            output_dir=output_dir,
        )


@hydra.main(config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_priors(cfg) -> None:
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
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    # Compute initial KSD
    ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    # Corner points optimization for prior
    if cfg.ksd.optimize:
        optimizer = OptimizationCornerPointsUnivariateGaussian(ksd_estimator, cfg.ksd.optimize.prior.Gaussian)
        qf_prior = optimizer.evaluate_all_prior_corners()
        qf_prior_all_combinations, corner_points = optimizer.evaluate_all_prior_combinations()

    # Corner points optimization for loss lr
    if cfg.ksd.optimize:
        optimizer = OptimizationCornerPointsUnivariateGaussian(ksd_estimator, cfg.ksd.optimize.loss.GaussianLogLikelihood)
        qf_lr = optimizer.evaluate_all_lr_corners()

    # Perform grid search over parameters box ranges if defined
    if cfg.ksd.optimize:
        plots_across_gaussian_prior_parameters_ranges(cfg, model, posterior_samples, kernel)
        plots_across_gaussian_parameters_ranges_etas_quadratic_form(cfg, qf_prior_all_combinations, corner_points)
        density_plot_across_gaussian_prior_parameter_set(cfg, model, posterior_samples, kernel)
        plots_across_gaussian_loss_lr_parameters_ranges(cfg, model, posterior_samples, kernel)


@hydra.main(config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_log_normal_priors(cfg) -> None:
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
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    # Compute initial KSD
    ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    if cfg.ksd.optimize:
        density_plot_across_prior_parameter_sets(cfg, model, posterior_samples, kernel)


@hydra.main(config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors(cfg) -> None:
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
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    # Compute initial KSD
    ksd_estimator = PosteriorKSD(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    # Corner points optimization for prior
    if cfg.ksd.optimize:
        optimizer = OptimizationCornerPointsMultivariateGaussian(ksd_estimator, cfg.ksd.optimize.prior.MultivariateGaussian, distribution_cls=MultivariateGaussian)
        qf_prior = optimizer.evaluate_all_prior_corners()
        qf_prior_all_combinations, corner_points = optimizer.evaluate_all_prior_combinations()

    if cfg.ksd.optimize:
        density_plot_across_multivariate_prior_parameter_sets(cfg, model, posterior_samples, kernel, qf_across_priors=qf_prior_all_combinations)


@hydra.main(config_path="../../configs/paper/ksd_calculation/toy/", config_name="inverse_wishart")
def run_inverse_wishart_priors(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    # Instantiate the Bayesian model with runtime overrides
    model = instantiate(cfg.model, data_config=cfg.data)

    # Sample from the posterior
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
    posterior_samples_vec = model.vectorize_samples(posterior_samples)

    # Instantiate the kernel
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples_vec)

    # Compute initial KSD
    ksd_estimator = PosteriorKSD(samples=posterior_samples_vec, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    # Corner points optimization
    if cfg.ksd.optimize:
        optimizer = OptimizationCornerPointsInverseWishart(ksd_estimator, cfg.ksd.optimize.prior.InverseWishart, distribution_cls=InverseWishart)
        results = optimizer.evaluate_all_corners()


if __name__ == "__main__":
    # run_gaussian_priors()
    # run_gaussian_log_normal_priors()
    run_multivariate_gaussian_priors()
    # run_inverse_wishart_priors()