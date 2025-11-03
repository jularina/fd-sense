from src.optimization.nonparametric_fisher import OptimisationNonparametricBase
from src.optimization.corner_points_fisher import *
from src.distributions.gaussian import MultivariateGaussian
from src.utils.files_operations import load_plot_config
from src.utils.distributions import DISTRIBUTION_MAP
from src.bayesian_model.base import BayesianModel
from src.plots.paper.toy_paper_fisher_funcs import *
from src.discrepancies.prior_fisher import PriorFDNonParametric
from src.discrepancies.posterior_fisher import PosteriorFDBase, PosteriorFDNonParametric
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
import time
import json

warnings.filterwarnings("ignore", category=UserWarning)


def density_plot_across_multivariate_prior_parameter_sets(
    cfg,
    model,
    posterior_samples,
    qf_priors_all_combinations,
):
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
    dist_results = {}
    all_dists = {}

    for param_dict in param_values:
        model.set_prior_parameters(param_dict, distribution_cls=distribution_cls)
        fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="prior")
        fisher = fisher_estimator.estimate_fisher()
        key = (tuple(param_dict["mu"].flatten()), tuple(param_dict["cov"].flatten()))
        dist_results[key] = fisher
        all_dists[key] = param_dict

        print(f"[INFO] Prior: {param_dict}, Fisher Divergence: {fisher:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_multivariate_priors_densities(
        all_params=all_dists,
        all_ksds=dist_results,
        output_dir=output_dir,
        plot_cfg=plot_cfg,
        true_theta=cfg.data.base_prior.mu,
        true_cov=cfg.data.base_prior.cov
    )
    plot_multivariate_joint_prior_densities_by_fd(
        results=qf_priors_all_combinations,
        output_dir=output_dir,
        plot_cfg=plot_cfg,
        true_theta=cfg.data.base_prior.mu,
        true_cov=cfg.data.base_prior.cov
    )


def plots_across_gaussian_prior_parameters_ranges(cfg, model: BayesianModel, posterior_samples: np.ndarray[float]):
    """
    Recalculates Fisher along all the possible hyperparameters combination across the ranges

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model loaded by Hydra.
        posterior_samples (np.ndarray[float]): Posterior samples
        kernel (BaseKernel): Kernel
    """
    results = {}
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
        fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="prior")
        fisher = fisher_estimator.estimate_fisher()
        results[tuple(values)] = fisher
        print(f"Prior: {prior_params}, mu_n: {model.mu_n}, Fisher Divergence: {fisher:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    param_names = [name+"_0" for name in param_names]
    plot_multi_line_plots(results, param_names, plot_cfg, output_dir)


def plots_across_gaussian_loss_lr_parameters_ranges(cfg, model: BayesianModel, posterior_samples: np.ndarray[float]):
    """
    Recalculates FD along all the possible hyperparameters combination across the ranges

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model loaded by Hydra.
        posterior_samples (np.ndarray[float]): Posterior samples
    """
    results = {}
    box_cfg = cfg.ksd.optimize.loss.GaussianLogLikelihood.parameters_box_range
    param_names = list(box_cfg.ranges.keys())
    param_ranges = [
        np.round(np.linspace(*box_cfg.ranges[name], num=box_cfg.nums[name]), 2)
        for name in param_names
    ]
    for values in np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(param_names)):
        params = dict(zip(param_names, values))
        model.set_lr_parameter(params["lr"])
        fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="loss")
        fisher = fisher_estimator.estimate_fisher()
        results[values[0]] = fisher
        print(f"Lr: {params}, FD: {fisher:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_single_param(results, param_names[0], plot_cfg, output_dir)


def plots_across_gaussian_parameters_ranges_etas_quadratic_form(cfg, eta_results, corner_points):
    """
    Recalculates FD along all the possible hyperparameters combination across the ranges

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model loaded by Hydra.
        posterior_samples (np.ndarray[float]): Posterior samples
    """
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_eta_surface(eta_results, corner_points, plot_cfg, output_dir)


def density_plot_across_univariate_prior_parameter_sets(cfg, model: BayesianModel, posterior_samples: np.ndarray):
    """
    Recalculates FD across all prior hyperparameter combinations for each distribution.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
        model (BayesianModel): Model.
        posterior_samples (np.ndarray): Posterior samples.
        kernel (BaseKernel): Kernel.
    """
    all_fd_results = {}
    param_values_dict = {"Gaussian": np.array([[-7, 1], [5, 3], [9, 2]]), "LogNormal": np.array([[1, 0.5]])}
    for dist_name, dist_cfg in cfg.ksd.optimize.prior.items():
        if dist_name not in DISTRIBUTION_MAP:
            continue
        distribution_cls = DISTRIBUTION_MAP[dist_name]
        box_cfg = dist_cfg.parameters_box_range
        param_names = list(box_cfg.ranges.keys())
        param_values = param_values_dict[dist_name]
        dist_fd_results = {}

        for values in param_values:
            prior_params = dict(zip(param_names, values))
            model.set_prior_parameters(prior_params, distribution_cls=distribution_cls)
            fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="prior")
            fd = fisher_estimator.estimate_fisher()
            dist_fd_results[tuple(values)] = fd
            print(f"Dist: {dist_name}, Prior: {prior_params}, mu_n: {model.mu_n}, FD: {fd:.4f}")

        all_fd_results[dist_name] = {
            "fd": dist_fd_results,
            "param_names": [p + "_0" for p in param_names],
            "distribution_cls": distribution_cls
        }

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_prior_densities_by_fd(
        all_ksd_data=all_fd_results,
        cfg=cfg,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="univariate_gaussian")
def run_gaussian_priors(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute Fisher divergence and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/univariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)

    posterior_samples = np.load(output_dir + "/posterior_samples.npy")
    fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="prior")
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher():.4f}")

    # Optimisation
    optimizer = OptimizationCornerPointsUnivariateGaussian(
        fisher_estimator, cfg.ksd.optimize.prior.Gaussian, cfg.ksd.optimize.loss.GaussianLogLikelihood)
    prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
    prior_combinations = optimizer.evaluate_all_prior_combinations()

    # plots_across_gaussian_prior_parameters_ranges(cfg, model, posterior_samples)
    plots_across_gaussian_parameters_ranges_etas_quadratic_form(cfg, prior_combinations, prior_corners)
    # density_plot_across_univariate_prior_parameter_sets(cfg, model, posterior_samples)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute Fisher and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)

    posterior_samples = np.load(output_dir + "/posterior_samples.npy")
    fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="prior")
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher():.4f}")

    optimizer = OptimizationCornerPointsMultivariateGaussian(
        fisher_estimator, cfg.ksd.optimize.prior.MultivariateGaussian,
        cfg.ksd.optimize.loss.MultivariateGaussianLogLikelihood)
    qf_priors_all_combinations = optimizer.evaluate_all_prior_combinations()

    density_plot_across_multivariate_prior_parameter_sets(
        cfg, model, posterior_samples, qf_priors_all_combinations=qf_priors_all_combinations)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_lr(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/univariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)

    posterior_samples = np.load(output_dir + "/posterior_samples.npy")
    fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="loss")
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher():.4f}")

    plots_across_gaussian_loss_lr_parameters_ranges(cfg, model, posterior_samples)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def run_gaussian_priors_nonparametric_diff_radii(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/univariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)
        np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

    posterior_samples = np.load(output_dir + "/posterior_samples.npy")
    prior_samples = np.load(output_dir + "/prior_samples.npy")

    fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="prior")
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher():.4f}")

    # Nonparametric optimisation
    estimator_prior = PriorFDNonParametric(samples=prior_samples, model=model, candidate_type="prior")
    estimator_posterior = PosteriorFDNonParametric(samples=posterior_samples, model=model, candidate_type="prior")
    psi_sdp_list, ksd_estimates_list, radius_labels = [], [], []

    for radius_lower_bound in [0.5, 2.0, 4.0, 6.0]:
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius_lower_bound=radius_lower_bound
        )
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        psi_sdp_list.append(result_sdp["psi_opt"])
        ksd_estimates_list.append(result_sdp["est"])
        radius_labels.append(radius_lower_bound)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

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


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_nonparametric_diff_radii(cfg, save_samples: bool = True) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)
        np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

    posterior_samples = np.load(output_dir + "/posterior_samples.npy")
    prior_samples = np.load(output_dir + "/prior_samples.npy")

    fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="prior")
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher():.4f}")
    psi_sdp_list, ksd_estimates_list, radius_labels = [], [], []

    # Nonparametric optimization
    estimator_prior = PriorFDNonParametric(samples=prior_samples, model=model, candidate_type="prior")
    estimator_posterior = PosteriorFDNonParametric(samples=posterior_samples, model=model, candidate_type="prior")

    for radius_lower_bound in [0.1, 0.5, 5.0, 10.0, 20]:
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius_lower_bound=radius_lower_bound
        )
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        psi_sdp_list.append(result_sdp["psi_opt"])
        ksd_estimates_list.append(result_sdp["est"])
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
        domain=((-10, 15), (-10, 15)),
        resolution=300
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_nonparametric_basis_funcs_nums(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/multivariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)
        np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

    posterior_samples = np.load(output_dir + "/posterior_samples.npy")
    prior_samples = np.load(output_dir + "/prior_samples.npy")

    fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="prior")
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher():.4f}")
    psi_sdp_list, ksd_estimates_list, radius_labels = [], [], []

    # Nonparametric optimization
    estimator_prior = PriorFDNonParametric(samples=prior_samples, model=model, candidate_type="prior")
    estimator_posterior = PosteriorFDNonParametric(samples=posterior_samples, model=model, candidate_type="prior")

    psi_sdp_list, ksd_estimates_list, radius_labels = [], [], []
    basis_funcs_nums = [5, 10, 15, 20]
    basis_list = []

    for basis_funcs_num in basis_funcs_nums:
        cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius_lower_bound=5.0
        )
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        psi_sdp_list.append(result_sdp["psi_opt"])
        ksd_estimates_list.append(result_sdp["est"])
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
        legend_title=plot_cfg.plot.param_latex_names.get("estimatedFDposteriorsShort", ""),
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=((-10, 15), (-10, 15)),
        resolution=300,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="inverse_wishart")
def run_inverse_wishart_priors(cfg) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.sample_posterior(cfg.data.posterior_samples_num)
    posterior_samples_vec = model.vectorize_samples(posterior_samples)
    fisher_estimator = PosteriorFDBase(samples=posterior_samples_vec, model=model, candidate_type="loss")
    print(f"Initial FD: {fisher_estimator.estimate_fisher():.4f}")

    optimizer = OptimizationCornerPointsInverseWishart(
        fisher_estimator, cfg.ksd.optimize.prior.InverseWishart, cfg.ksd.optimize.loss.MultivariateGaussianLogLikelihood)
    qf_prior_all_combinations = optimizer.evaluate_all_prior_combinations()

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_inverse_wishart_scale_ellipses_by_fd_one_subplot(qf_prior_all_combinations, output_dir, plot_cfg)


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
    steps = 30

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            cfg.data.posterior_samples_num = sample_nums
            model = instantiate(cfg.model, data_config=cfg.data)
            start = time.perf_counter()
            fisher_estimator = PosteriorFDBase(samples=model.posterior_samples_init,
                                               model=model, candidate_type="prior")
            optimizer = OptimizationCornerPointsUnivariateGaussian(
                fisher_estimator, cfg.ksd.optimize.prior.Gaussian, cfg.ksd.optimize.loss.GaussianLogLikelihood)
            prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
            elapsed = time.perf_counter() - start
            largest_fd = prior_corners[0][2]
            times_list_parametric.append((sample_nums, elapsed))
            times_parametric[sample_nums][step] = elapsed
            print(f"***Parametric*** Samples: {sample_nums}, Initial FD: {largest_fd:.4f}, Time: {elapsed:.3f} sec")

    data_path = os.path.join(get_original_cwd(), "data/univariate_gaussian/runtimes/")
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
                start = time.perf_counter()

                estimator_prior = PriorFDNonParametric(
                    samples=model.prior_samples_init, model=model, candidate_type="prior")
                estimator_posterior = PosteriorFDNonParametric(
                    samples=model.posterior_samples_init, model=model, candidate_type="prior")
                optimizer = OptimisationNonparametricBase(
                    estimator_posterior,
                    estimator_prior,
                    cfg.ksd.optimize.prior.nonparametric,
                    radius_lower_bound=5.0
                )
                result_sdp = optimizer.optimize_through_sdp_relaxation()
                elapsed = time.perf_counter() - start
                largest_ksd = result_sdp["est"]
                times_list_nonparametric.append((sample_nums, basis_funcs_num, elapsed))
                times_nonparametric[sample_nums][basis_funcs_num][step] = elapsed
                print(
                    f"***Non-parametric*** Samples: {sample_nums}, Basis Functions num: {basis_funcs_num}, Initial FD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    with open(data_path + "nonparametric_optimisation_times.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)


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

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            cfg.data.posterior_samples_num = sample_nums
            model = instantiate(cfg.model, data_config=cfg.data)
            start = time.perf_counter()
            fisher_estimator = PosteriorFDBase(samples=model.posterior_samples_init,
                                               model=model, candidate_type="prior")
            optimizer = OptimizationCornerPointsMultivariateGaussian(
                fisher_estimator, cfg.ksd.optimize.prior.MultivariateGaussian,
                cfg.ksd.optimize.loss.MultivariateGaussianLogLikelihood)
            prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
            elapsed = time.perf_counter() - start
            largest_fd = prior_corners[0][2]
            times_list_parametric.append((sample_nums, elapsed))
            times_parametric[sample_nums][step] = elapsed
            print(f"***Parametric*** Samples: {sample_nums}, Initial FD: {largest_fd:.4f}, Time: {elapsed:.3f} sec")

    data_path = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/")
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
                start = time.perf_counter()
                estimator_prior = PriorFDNonParametric(samples=model.prior_samples_init, model=model,
                                                       candidate_type="prior")
                estimator_posterior = PosteriorFDNonParametric(samples=model.posterior_samples_init, model=model,
                                                               candidate_type="prior")
                optimizer = OptimisationNonparametricBase(
                    estimator_posterior,
                    estimator_prior,
                    cfg.ksd.optimize.prior.nonparametric,
                    radius_lower_bound=5.0
                )
                result_sdp = optimizer.optimize_through_sdp_relaxation()
                elapsed = time.perf_counter() - start
                largest_ksd = result_sdp["est"]
                times_list_nonparametric.append((sample_nums, basis_funcs_num, elapsed))
                times_nonparametric[sample_nums][basis_funcs_num][step] = elapsed
                print(
                    f"***Non-parametric*** Samples: {sample_nums}, Basis Functions num: {basis_funcs_num}, Initial FD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    with open(data_path + "nonparametric_optimisation_times.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_diff_basis_funcs_num(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    times_list_parametric, times_list_nonparametric = [], []
    basis_funcs_num_list = [int(x) for x in np.linspace(5, 25, 21)]
    times_nonparametric = defaultdict(dict)
    steps = 10

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for basis_funcs_num in basis_funcs_num_list:
            cfg.data.posterior_samples_num = 2000
            cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
            model = instantiate(cfg.model, data_config=cfg.data)
            start = time.perf_counter()
            estimator_prior = PriorFDNonParametric(samples=model.prior_samples_init, model=model,
                                                   candidate_type="prior")
            estimator_posterior = PosteriorFDNonParametric(samples=model.posterior_samples_init, model=model,
                                                           candidate_type="prior")
            optimizer = OptimisationNonparametricBase(
                estimator_posterior,
                estimator_prior,
                cfg.ksd.optimize.prior.nonparametric,
                radius_lower_bound=5.0
            )
            result_sdp = optimizer.optimize_through_sdp_relaxation()
            elapsed = time.perf_counter() - start
            largest_ksd = result_sdp["est"]
            times_list_nonparametric.append((basis_funcs_num, elapsed))
            times_nonparametric[basis_funcs_num][step] = elapsed
            print(
                f"***Non-parametric*** Basis Functions num: {basis_funcs_num}, Initial FD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    data_path = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/")
    with open(data_path + "nonparametric_optimisation_times_diff_basis_funcs_nums.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_priors_optimisation_runtimes(cfg, dim: str = "multivariate"):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    data_path = os.path.join(get_original_cwd(), f"data/{dim}_gaussian/runtimes/")
    with open(data_path + "parametric_optimisation_times.json", "r") as f:
        parametric_optimisation_times = json.load(f)

    with open(data_path + "nonparametric_optimisation_times.json", "r") as f:
        nonparametric_optimisation_times = json.load(f)

    plot_runtime_parametric_nonparametric_with_ci(
        parametric_optimisation_times,
        nonparametric_optimisation_times,
        plot_cfg,
        output_dir,
        filename=f"runtime_parametric_nonparametric_{dim}.pdf"
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_priors_optimisation_runtimes(cfg):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    data_path = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/")

    with open(data_path + "nonparametric_optimisation_times_diff_basis_funcs_nums.json", "r") as f:
        nonparametric_optimisation_times = json.load(f)

    plot_runtime_nonparametric_with_ci(
        nonparametric_optimisation_times,
        plot_cfg,
        output_dir,
        filename="runtime_nonparametric_multivariate.pdf"
    )


if __name__ == "__main__":
    # run_gaussian_priors()
    # run_gaussian_lr()
    run_multivariate_gaussian_priors()
    # run_inverse_wishart_priors()
    # run_gaussian_priors_nonparametric_diff_radii()
    # run_multivariate_gaussian_priors_nonparametric_diff_radii()
    # run_multivariate_gaussian_priors_nonparametric_basis_funcs_nums()
    # run_gaussian_priors_diff_samples_num()
    # run_multivariate_gaussian_priors_diff_samples_num()
    # run_multivariate_gaussian_priors_diff_basis_funcs_num()
    # run_priors_optimisation_runtimes()
