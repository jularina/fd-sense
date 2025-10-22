import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd

warnings.filterwarnings("ignore", category=UserWarning)

from src.discrepancies.posterior_fisher import PosteriorFDBase, PosteriorFDNonParametric
from src.discrepancies.prior_fisher import PriorFDNonParametric
from src.plots.paper.toy_paper_fisher_funcs import *
from src.bayesian_model.base import BayesianModel
from src.utils.distributions import DISTRIBUTION_MAP
from src.utils.files_operations import load_plot_config
from src.distributions.gaussian import MultivariateGaussian
from src.optimization.corner_points_fisher import OptimizationCornerPointsUnivariateGaussian
from src.optimization.nonparametric_fisher import OptimisationNonparametricBase


def density_plot_across_multivariate_prior_parameter_sets(
    cfg,
    model,
    posterior_samples,
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
    optimizer = OptimizationCornerPointsUnivariateGaussian(fisher_estimator, cfg.ksd.optimize.prior.Gaussian, cfg.ksd.optimize.loss.GaussianLogLikelihood)
    prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
    prior_combinations = optimizer.evaluate_all_prior_combinations()

    # plots_across_gaussian_prior_parameters_ranges(cfg, model, posterior_samples)
    plots_across_gaussian_parameters_ranges_etas_quadratic_form(cfg, prior_combinations, prior_corners)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors(cfg, save_samples: bool = True) -> None:
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

    density_plot_across_multivariate_prior_parameter_sets(cfg, model, posterior_samples)

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


if __name__ == "__main__":
    # run_gaussian_priors()
    # run_gaussian_lr()
    # run_multivariate_gaussian_priors()
    run_gaussian_priors_nonparametric_diff_radii()
