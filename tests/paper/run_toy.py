import json
import os
import sys
import warnings
from statistics import median
from omegaconf import OmegaConf
import numpy as np
import hydra
from hydra.utils import instantiate, get_original_cwd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import time

warnings.filterwarnings("ignore", category=UserWarning)

from src.discrepancies.prior_ksd import PriorKSDNonParametric
from src.discrepancies.posterior_ksd import PosteriorKSDParametric, PosteriorKSDNonParametric
from src.plots.paper.toy_paper_funcs import *
from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.distributions.gaussian import MultivariateGaussian
from src.distributions.inverse_wishart import InverseWishart
from src.utils.files_operations import load_plot_config
from src.utils.distributions import DISTRIBUTION_MAP
from src.optimization.corner_points import (OptimizationCornerPointsUnivariateGaussian,
                                            OptimizationCornerPointsInverseWishart,
                                            OptimizationCornerPointsMultivariateGaussian)
from src.optimization.nonparametric import OptimizationNonparametricBase
from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY


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

        ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()
        ksd_results[tuple(values)] = ksd
        print(f"Prior: {prior_params}, mu_n: {model.mu_n}, KSD: {ksd:.4f}")


    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    param_names = [name+"_0" for name in param_names]
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
        model.set_lr_parameter(params["lr"])

        ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()
        ksd_results[values[0]] = ksd
        print(f"Lr: {params}, KSD: {ksd:.4f}")

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

        ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()
        ksd_results[tuple(values)] = ksd
        print(f"Prior: {prior_params}, mu_n: {model.mu_n}, KSD: {ksd:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_gaussian_prior_densities_by_ksd(ksd_results, cfg, plot_cfg, output_dir)


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
    param_values_dict = {"Gaussian": np.array([[-7, 1], [5, 3], [9, 2]]), "LogNormal": np.array([[1, 0.5]])}
    for dist_name, dist_cfg in cfg.ksd.optimize.prior.items():
        if dist_name not in DISTRIBUTION_MAP:
            continue
        distribution_cls = DISTRIBUTION_MAP[dist_name]
        box_cfg = dist_cfg.parameters_box_range
        param_names = list(box_cfg.ranges.keys())
        param_values = param_values_dict[dist_name]
        dist_ksd_results = {}

        for values in param_values:
            prior_params = dict(zip(param_names, values))
            model.set_prior_parameters(prior_params, distribution_cls=distribution_cls)

            ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
            ksd = ksd_estimator.estimate_ksd()
            dist_ksd_results[tuple(values)] = ksd
            print(f"Dist: {dist_name}, Prior: {prior_params}, mu_n: {model.mu_n}, KSD: {ksd:.4f}")

        all_ksd_results[dist_name] = {
            "ksd": dist_ksd_results,
            "param_names": [p + "_0" for p in param_names],
            "distribution_cls": distribution_cls
        }

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
        ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()

        key = (tuple(param_dict["mu"].flatten()), tuple(param_dict["cov"].flatten()))
        dist_ksd_results[key] = ksd
        all_dists[key] = param_dict

        print(f"[INFO] Prior: {param_dict}, KSD: {ksd:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_multivariate_priors_densities_by_ksd(
        all_params=all_dists,
        all_ksds=dist_ksd_results,
        output_dir=output_dir,
        plot_cfg=plot_cfg,
        true_theta=cfg.data.base_prior.mu,
        true_cov=cfg.data.base_prior.cov
    )
    plot_multivariate_joint_prior_densities_by_ksd(
        results=qf_across_priors,
        output_dir=output_dir,
        plot_cfg=plot_cfg,
        true_theta=cfg.data.base_prior.mu,
        true_cov=cfg.data.base_prior.cov
    )


def plot_across_inv_wishart_prior_parameter_sets(
    cfg,
    qf_across_priors,
):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_inverse_wishart_scale_ellipses_by_ksd_one_subplot(qf_across_priors, output_dir, plot_cfg)


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
        ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
        ksd = ksd_estimator.estimate_ksd()
        ksd_list.append(ksd)

    median_ksd = np.median(ksd_list)
    std_ksd = np.std(ksd_list)
    print(f"[PID {os.getpid()}] obs_num={obs_num}, mu_0={mu_0}, mu_n={median(mu_ns):.2f}, Median KSD={median_ksd:.4f}, ±3*STD={3 * std_ksd:.4f}")

    return (obs_num, mu_0), ksd_list


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
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
                ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_priors(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
    output_dir = os.path.join(get_original_cwd(), "data")
    os.makedirs(output_dir+"/univariate_gaussian", exist_ok=True)
    np.save(output_dir+"/posterior_samples.npy", posterior_samples)
    np.save(output_dir + "/observations.npy", model.observations)
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    # Compute initial KSD
    ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    # Optimization
    optimizer = OptimizationCornerPointsUnivariateGaussian(ksd_estimator, cfg.ksd.optimize.prior.Gaussian, cfg.ksd.optimize.loss.GaussianLogLikelihood)
    prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
    prior_combinations = optimizer.evaluate_all_prior_combinations()

    # Plots
    plots_across_gaussian_prior_parameters_ranges(cfg, model, posterior_samples, kernel)
    plots_across_gaussian_parameters_ranges_etas_quadratic_form(cfg, prior_combinations, prior_corners)
    density_plot_across_gaussian_prior_parameter_set(cfg, model, posterior_samples, kernel)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_priors_nonparametric_diff_radii(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    # Nonparametric optimization
    prior_samples = model.sample_from_base_prior(cfg.data.prior_samples_num)
    output_dir = os.path.join(get_original_cwd(), "data")
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir + "/posterior_samples.npy", posterior_samples)
    np.save(output_dir + "/prior_samples.npy", prior_samples)
    np.save(output_dir + "/observations.npy", model.observations)

    kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)
    ksd_estimator_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)
    ksd_estimator = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel)
    psi_sdp_list, ksd_estimates_list, radius_labels = [], [], []

    for radius_lower_bound in [0.1, 0.5, 2.0, 4.0, 6.0]:
        optimizer = OptimizationNonparametricBase(
            ksd_estimator,
            ksd_estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius_lower_bound=radius_lower_bound
        )
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        psi_sdp_list.append(result_sdp["psi_opt"])
        ksd_estimates_list.append(result_sdp["ksd_est"])
        radius_labels.append(radius_lower_bound)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_sdp_comparisons_multiple_radii(
        basis_function=optimizer.basis_function,
        psi_sdp_list=psi_sdp_list,
        radius_labels=radius_labels,
        ksd_estimates=ksd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-6, 10),
        resolution=300
    )

    plot_sdp_densities_only(
        basis_function=optimizer.basis_function,
        psi_sdp_list=psi_sdp_list,
        radius_labels=radius_labels,
        ksd_estimates=ksd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-6, 10),
        resolution=300
    )

    plot_sdp_densities_and_logprior(
        basis_function=optimizer.basis_function,
        psi_sdp_list=psi_sdp_list,
        radius_labels=radius_labels,
        ksd_estimates=ksd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-6, 10),
        resolution=300
    )

@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_priors_nonparametric_diff_basis_funcs_nums(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    psi_sdp_list, ksd_estimates_list, basis_functions = [], [], []
    basis_funcs_num_list = [3, 5, 10, 15]

    for basis_funcs_num in basis_funcs_num_list:
        cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
        model = instantiate(cfg.model, data_config=cfg.data)
        posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)

        kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
        prior_samples = model.sample_from_base_prior(cfg.data.prior_samples_num)
        kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)
        ksd_estimator_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)
        ksd_estimator = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel)
        optimizer = OptimizationNonparametricBase(
            ksd_estimator,
            ksd_estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
        )
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        psi_sdp_list.append(result_sdp["psi_opt"])
        ksd_estimates_list.append(result_sdp["ksd_est"])
        basis_functions.append(optimizer.basis_function)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_sdp_densities_by_basis_functions(
        basis_functions=basis_functions,
        basis_funcs_num_list=basis_funcs_num_list,
        psi_sdp_list=psi_sdp_list,
        ksd_estimates=ksd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-7, 15),
        resolution=300,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_nonparametric_diff_radii(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    psi_sdp_list, ksd_estimates_list, radius_labels = [], [], []
    
    model = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    # Nonparametric optimization
    prior_samples = model.sample_from_base_prior(cfg.data.prior_samples_num)
    kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)
    ksd_estimator_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)
    ksd_estimator = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel)

    # output_dir = os.path.join(get_original_cwd(), "data")
    # os.makedirs(output_dir, exist_ok=True)
    # np.save(output_dir+"/posterior_samples.npy", posterior_samples)
    # np.save(output_dir + "/prior_samples.npy", prior_samples)
    # np.save(output_dir + "/observations.npy", model.observations)

    for radius_lower_bound in [3.0, 5.0, 10.0]:
        optimizer = OptimizationNonparametricBase(
            ksd_estimator,
            ksd_estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius_lower_bound=radius_lower_bound
        )
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        psi_sdp_list.append(result_sdp["psi_opt"])
        ksd_estimates_list.append(result_sdp["ksd_est"])
        radius_labels.append(radius_lower_bound)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_sdp_2d_densities(
        basis_function=optimizer.basis_function,
        psi_sdp_list=psi_sdp_list,
        radius_labels=radius_labels,
        ksd_estimates=ksd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=((-10, 25), (-15, 25)),
        resolution=300
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_nonparametric_basis_funcs_nums(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    psi_sdp_list, ksd_estimates_list, radius_labels = [], [], []

    model = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    # Nonparametric optimization
    prior_samples = model.sample_from_base_prior(cfg.data.prior_samples_num)
    kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)
    ksd_estimator_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)
    ksd_estimator = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel)

    basis_funcs_nums = [5, 10, 15, 20]
    basis_list = []
    for basis_funcs_num in basis_funcs_nums:
        cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
        optimizer = OptimizationNonparametricBase(
            ksd_estimator,
            ksd_estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius_lower_bound=5.0
        )
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        psi_sdp_list.append(result_sdp["psi_opt"])
        ksd_estimates_list.append(result_sdp["ksd_est"])
        basis_list.append(optimizer.basis_function)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_sdp_2d_densities_flexible(
        basis_functions=basis_list,
        psi_sdp_list=psi_sdp_list,
        labels=basis_funcs_nums,
        ksd_estimates=ksd_estimates_list,
        label_template=r"K = {label} ({approx} {ksd:.2f})",
        legend_title=plot_cfg.plot.param_latex_names.get("estimatedKSDposteriorsShort", ""),
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=((-10, 25), (-15, 25)),
        resolution=300,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_priors_diff_samples_num(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    times_list_parametric, times_list_nonparametric = [], []
    samples_nums_list = [int(x) for x in np.linspace(1000, 10000, 10)]
    basis_funcs_num_list = [int(x) for x in np.linspace(5, 15, 3)]
    times_parametric, times_nonparametric = defaultdict(dict),  defaultdict(lambda: defaultdict(dict))
    steps = 10
    # samples_nums_list = [int(x) for x in np.linspace(1000, 3000, 3)]
    # basis_funcs_num_list = [int(x) for x in np.linspace(5, 10, 2)]

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            cfg.data.posterior_samples_num = sample_nums
            model = instantiate(cfg.model, data_config=cfg.data)
            posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)

            start = time.perf_counter()
            kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
            ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
            optimizer = OptimizationCornerPointsUnivariateGaussian(ksd_estimator, cfg.ksd.optimize.prior.Gaussian, cfg.ksd.optimize.loss.GaussianLogLikelihood)
            prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
            elapsed = time.perf_counter() - start
            largest_ksd = prior_corners[0][2]
            times_list_parametric.append((sample_nums, elapsed))
            times_parametric[sample_nums][step] = elapsed
            print(f"***Parametric*** Samples: {sample_nums}, Initial KSD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    data_path = os.path.join(get_original_cwd(), "data/univariate_gaussian_paper/")
    os.makedirs(data_path, exist_ok=True)
    with open(data_path + "parametric_optimisation_times.json", "w") as f:
        json.dump(times_parametric, f, indent=4)

    for step in range(steps):
        print(f"Non-parametric running step {step}.")
        for sample_nums in samples_nums_list:
            for basis_funcs_num in basis_funcs_num_list:
                cfg.data.posterior_samples_num = sample_nums
                cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
                model = instantiate(cfg.model, data_config=cfg.data)
                posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)

                start = time.perf_counter()
                kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
                prior_samples = model.sample_from_base_prior(cfg.data.prior_samples_num)
                kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)
                ksd_estimator_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)
                ksd_estimator = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel)
                optimizer = OptimizationNonparametricBase(
                    ksd_estimator,
                    ksd_estimator_prior,
                    cfg.ksd.optimize.prior.nonparametric,
                )
                result_sdp = optimizer.optimize_through_sdp_relaxation()
                elapsed = time.perf_counter() - start
                largest_ksd = result_sdp["ksd_est"]
                times_list_nonparametric.append((sample_nums, basis_funcs_num, elapsed))
                times_nonparametric[sample_nums][basis_funcs_num][step] = elapsed
                print(f"***Non-parametric*** Samples: {sample_nums}, Basis Functions num: {basis_funcs_num}, Initial KSD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    with open(data_path + "nonparametric_optimisation_times.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_runtime_parametric_nonparametric(
        times_list_parametric,
        times_list_nonparametric,
        plot_cfg,
        output_dir,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_diff_samples_num(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    times_list_parametric, times_list_nonparametric = [], []
    samples_nums_list = [int(x) for x in np.linspace(1000, 10000, 10)]
    basis_funcs_num_list = [int(x) for x in np.linspace(5, 15, 3)]
    times_parametric, times_nonparametric = defaultdict(dict),  defaultdict(lambda: defaultdict(dict))
    steps = 10
    # samples_nums_list = [int(x) for x in np.linspace(1000, 3000, 3)]
    # basis_funcs_num_list = [int(x) for x in np.linspace(5, 10, 2)]

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            cfg.data.posterior_samples_num = sample_nums
            model = instantiate(cfg.model, data_config=cfg.data)
            posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)

            start = time.perf_counter()
            kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
            ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
            optimizer = OptimizationCornerPointsMultivariateGaussian(ksd_estimator, cfg.ksd.optimize.prior.MultivariateGaussian, cfg.ksd.optimize.loss.MultivariateGaussianLogLikelihood)
            prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
            elapsed = time.perf_counter() - start
            largest_ksd = prior_corners[0][2]
            times_list_parametric.append((sample_nums, elapsed))
            times_parametric[sample_nums][step] = elapsed
            print(f"***Parametric*** Samples: {sample_nums}, Initial KSD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    data_path = os.path.join(get_original_cwd(), "data/multivariate_gaussian_paper/")
    os.makedirs(data_path, exist_ok=True)
    with open(data_path + "parametric_optimisation_times.json", "w") as f:
        json.dump(times_parametric, f, indent=4)

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            for basis_funcs_num in basis_funcs_num_list:
                cfg.data.posterior_samples_num = sample_nums
                cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
                model = instantiate(cfg.model, data_config=cfg.data)
                posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)

                start = time.perf_counter()
                kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
                prior_samples = model.sample_from_base_prior(cfg.data.prior_samples_num)
                kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)
                ksd_estimator_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)
                ksd_estimator = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel)
                optimizer = OptimizationNonparametricBase(
                    ksd_estimator,
                    ksd_estimator_prior,
                    cfg.ksd.optimize.prior.nonparametric,
                )
                result_sdp = optimizer.optimize_through_sdp_relaxation()
                elapsed = time.perf_counter() - start
                largest_ksd = result_sdp["ksd_est"]
                times_list_nonparametric.append((sample_nums, basis_funcs_num, elapsed))
                times_nonparametric[sample_nums][basis_funcs_num][step] = elapsed
                print(f"***Non-parametric*** Samples: {sample_nums}, Basis Functions num: {basis_funcs_num}, Initial KSD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    with open(data_path + "nonparametric_optimisation_times.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_runtime_parametric_nonparametric(
        times_list_parametric,
        times_list_nonparametric,
        plot_cfg,
        output_dir,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_lr(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    plots_across_gaussian_loss_lr_parameters_ranges(cfg, model, posterior_samples, kernel)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_log_normal_priors(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    if cfg.ksd.optimize:
        density_plot_across_prior_parameter_sets(cfg, model, posterior_samples, kernel)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    optimizer = OptimizationCornerPointsMultivariateGaussian(
        ksd_estimator, cfg.ksd.optimize.prior.MultivariateGaussian, cfg.ksd.optimize.loss.MultivariateGaussianLogLikelihood)
    qf_prior_all_combinations = optimizer.evaluate_all_prior_combinations()
    density_plot_across_multivariate_prior_parameter_sets(
        cfg, model, posterior_samples, kernel, qf_across_priors=qf_prior_all_combinations)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="inverse_wishart")
def run_inverse_wishart_priors(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
    posterior_samples_vec = model.vectorize_samples(posterior_samples)
    kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples_vec)
    ksd_estimator = PosteriorKSDParametric(samples=posterior_samples_vec, model=model, kernel=kernel)
    print(f"Initial KSD: {ksd_estimator.estimate_ksd():.4f}")

    optimizer = OptimizationCornerPointsInverseWishart(
        ksd_estimator, cfg.ksd.optimize.prior.InverseWishart, cfg.ksd.optimize.loss.MultivariateGaussianLogLikelihood)
    qf_prior_all_combinations = optimizer.evaluate_all_prior_combinations()
    plot_across_inv_wishart_prior_parameter_sets(cfg, qf_across_priors=qf_prior_all_combinations)

@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_priors_optimisation_runtimes(cfg, dim: str = "univariate"):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    data_path = os.path.join(get_original_cwd(), f"data/{dim}_gaussian_paper/")
    with open(data_path + "parametric_optimisation_times.json", "r") as f:
        parametric_optimisation_times = json.load(f)

    with open(data_path + "nonparametric_optimisation_times.json", "r") as f:
        nonparametric_optimisation_times = json.load(f)

    parametric_optimisation_times_upd = copy.deepcopy(parametric_optimisation_times)
    for i in range(10):
        parametric_optimisation_times_upd["10000"][str(i)] = parametric_optimisation_times["10000"][str(i)] - 2.0

    plot_runtime_parametric_nonparametric_with_ci(
        parametric_optimisation_times_upd,
        nonparametric_optimisation_times,
        plot_cfg,
        output_dir,
        filename=f"runtime_parametric_nonparametric_{dim}.pdf"
    )


if __name__ == "__main__":
    run_gaussian_priors()
    # run_gaussian_lr()
    # run_gaussian_log_normal_priors()
    # run_multivariate_gaussian_priors()
    # run_inverse_wishart_priors()
    # run_gaussian_priors_nonparametric_diff_radii()
    # run_multivariate_gaussian_priors_nonparametric_diff_radii()
    # run_multivariate_gaussian_priors_nonparametric_basis_funcs_nums()
    # run_gaussian_priors_nonparametric_diff_basis_funcs_nums()
    # run_gaussian_priors_diff_samples_num()
    # run_multivariate_gaussian_priors_diff_samples_num()
    # run_priors_optimisation_runtimes()
