from src.optimization.nonparametric_fisher import OptimisationNonparametricBase
from src.optimization.qcqp import ParametricQCQPBase
from src.optimization.corner_points_fisher import *
from src.utils.files_operations import *
from src.plots.paper.toy_paper_fisher_funcs import *
from src.discrepancies.prior_fisher import PriorFDParametric, PriorFDNonParametric
from src.discrepancies.posterior_fisher import PosteriorFDParametric, PosteriorFDNonParametric

import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
import time
import json

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
            config_name="univariate_gaussian")
def run_gaussian_priors_qcqp(cfg) -> None:
    """
    Compute Fisher divergence and optimize parametrically with QCQP.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    model = instantiate(cfg.model, data_config=cfg.data)

    # Prior
    prior_fd = PriorFDParametric(model=model)
    print(f"FD from score differences: {prior_fd.estimate_fisher_prior_only():.4f}")
    A_c, b_c, c_c = prior_fd.compute_fisher_quadratic_form_prior_only()
    eta = model.prior_candidate.natural_parameters()
    print(f"FD from quadratic form: {eta @ A_c @ eta + b_c @ eta + c_c:.4f}")

    # Posterior
    posterior_fd = PosteriorFDParametric(model)
    # Prior-only
    A, b, c = posterior_fd.compute_fisher_quadratic_form_prior_only()
    eta = model.prior_candidate.natural_parameters()
    # print("Prior-only: score diff:", posterior_fd.estimate_fisher_prior_only())
    # print("Prior-only: quadratic :", eta @ A @ eta + b @ eta + c)
    # LR-only
    A_lr, b_lr, c_lr = posterior_fd.compute_fisher_quadratic_form_lr_only()
    beta = model.loss_lr
    # print("LR-only: score diff:", posterior_fd.estimate_fisher_lr_only())
    # print("LR-only: quadratic :", A_lr * beta ** 2 + b_lr * beta + c_lr)

    # Radius choice
    eta_ref = model.prior_init.natural_parameters()
    min_r = eta_ref @ A_c @ eta_ref + eta_ref @ b_c + c_c

    # QCQP Optimisation
    solver = ParametricQCQPBase(posterior_fd, prior_fd)
    solution = solver.solve_generalized_eigenvalue(r=1, check_kernel_condition=True)
    print("lambda_star:", solution.lambda_star)
    print("eta_star:", solution.eta_star)
    print("constraint x^T A_c x:", solution.achieved_constraint)
    print("objective  x^T A x  :", solution.achieved_objective)

    print("SDP lambda t dual")
    sdp_lambda_t_dual_solution = solver.solve_dual_sdp_lambda_t(radius=1)
    print("lambda_star:", sdp_lambda_t_dual_solution.lambda_star)
    print("eta_star:", sdp_lambda_t_dual_solution.eta_star)
    print("dual_value:", sdp_lambda_t_dual_solution.dual_value)
    print("objective at eta_star:", sdp_lambda_t_dual_solution.primal_value)
    print("constraint at eta_star:", sdp_lambda_t_dual_solution.constraint_value, "(should be <= r)")

    print("Lagrange dual")
    lagrange_dual_solution = solver.solve_dual_1d_lambda(radius=1.0)
    print("eta_star:", lagrange_dual_solution.eta_star)
    print("lambda_star:", lagrange_dual_solution.lambda_star)
    print("dual_value:", lagrange_dual_solution.dual_value)
    print("objective at eta_star:", lagrange_dual_solution.primal_value)
    print("constraint at eta_star:", lagrange_dual_solution.constraint_value, "(should be <= r)")

    print("SDP relaxation")
    sdp_dual_solution = solver.solve_primal_sdp_relaxation(radius=1)
    print("lambda_star:", sdp_dual_solution.lambda_star)
    print("eta_star:", sdp_dual_solution.eta_star)
    print("dual_value:", sdp_dual_solution.dual_value)
    print("objective at eta_star:", sdp_dual_solution.primal_value)
    print("constraint at eta_star:", sdp_dual_solution.constraint_value, "(should be <= r)")

@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    optimizer = OptimisationNonparametricBase(
        estimator_posterior,
        estimator_prior,
        cfg.optimize.nonparametric,
        radius=5.0
    )
    start = time.perf_counter()
    result_sdp = optimizer.optimize_through_sdp_relaxation()
    elapsed = time.perf_counter() - start
    print(f"SDP primal relaxation time: {elapsed}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_prior_neighbourhood_comparison(
        optimizer=optimizer,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-10, 12),
        resolution=500,
        epsilon=0.2,
        n_nonparam_samples=40,
        mu_range=(-2.0, 6.0),
        sigma_range=(2.0, 8.0),
        n_param_grid=12,
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian_nonparam")
def run_gaussian_priors_nonparametric_diff_radii(cfg, save_samples: bool = False) -> None:
    """
    Main function to compute FD and perform prior parameter grid search using Hydra for configuration.
    """
    model = instantiate(cfg.model, data_config=cfg.data)
    output_dir = os.path.join(get_original_cwd(), "data/univariate_gaussian")

    if save_samples:
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + "/posterior_samples.npy", model.posterior_samples_init)
        np.save(output_dir + "/observations.npy", model.observations)
        np.save(output_dir + "/prior_samples.npy", model.prior_samples_init)

    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    sdp_lambda_list, sdp_fd_estimates_list, radius_labels = [], [], []
    sdp_lambda_list, sdp_fd_estimates_list = [], []

    for radius in [0.5, 1.0, 5.0, 10.0]:
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.optimize.nonparametric,
            radius=radius
        )
        start = time.perf_counter()
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        elapsed = time.perf_counter() - start
        print(f"SDP primal relaxation time: {elapsed}")

        start = time.perf_counter()
        result_lagrange_dual = optimizer.optimize_through_dual_1d_lambda()
        elapsed = time.perf_counter() - start
        print(f"Lagrange dual time: {elapsed}")

        start = time.perf_counter()
        result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
        elapsed = time.perf_counter() - start
        print(f"SDP dual time: {elapsed}")

        sdp_lambda_list.append(result_sdp["lambda_star"])
        sdp_fd_estimates_list.append(result_sdp["primal_value"])
        radius_labels.append(radius)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_sdp_densities_and_logprior(
        basis_function=optimizer.basis_function,
        sdp_lambda_list=sdp_lambda_list,
        radius_labels=radius_labels,
        estimates=sdp_fd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=(-10, 12),
        resolution=500
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
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

    # Nonparametric optimisation
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    sdp_lambda_list, fd_estimates_list, radius_labels = [], [], []

    for radius in [0.5, 1.0, 5.0, 10.0]:
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius=radius
        )
        start = time.perf_counter()
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        elapsed = time.perf_counter() - start
        print(f"SDP relaxation time: {elapsed}")

        sdp_lambda_list.append(result_sdp["lambda_star"])
        fd_estimates_list.append(result_sdp["primal_value"])
        radius_labels.append(radius)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_sdp_2d_densities(
        basis_function=optimizer.basis_function,
        sdp_lambda_list=sdp_lambda_list,
        radius_labels=radius_labels,
        ksd_estimates=fd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=((-10, 15), (-10, 15)),
        resolution=300
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/",
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

    fisher_estimator = PosteriorFDNonParametric(samples=posterior_samples, model=model, candidate_type="prior")
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher():.4f}")
    sdp_lambda_list, ksd_estimates_list, radius_labels = [], [], []

    # Nonparametric optimization
    estimator_prior = PriorFDNonParametric(samples=prior_samples, model=model, candidate_type="prior")
    estimator_posterior = PosteriorFDNonParametric(samples=posterior_samples, model=model, candidate_type="prior")

    sdp_lambda_list, ksd_estimates_list, radius_labels = [], [], []
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
        start = time.perf_counter()
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        elapsed = time.perf_counter() - start
        print(f"SDP time: {elapsed}")

        start = time.perf_counter()
        result_tr = optimizer.optimize_through_qcqp_trust_region()
        elapsed = time.perf_counter() - start
        print(f"TR time: {elapsed}")

        sdp_lambda_list.append(result_sdp["lambda_star"])
        ksd_estimates_list.append(result_sdp["est"])
        basis_list.append(optimizer.basis_function)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_sdp_2d_densities_flexible(
        basis_functions=basis_list,
        sdp_lambda_list=sdp_lambda_list,
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

@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="univariate_gaussian")
def run_gaussian_priors_nonparametric_diff_samples_num(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    times_list_parametric, times_list_nonparametric = [], []
    samples_nums_list = [int(x) for x in np.linspace(1000, 10000, 10)]
    basis_funcs_num_list = [int(x) for x in np.linspace(5, 15, 3)]
    times_parametric, times_nonparametric = defaultdict(dict),  defaultdict(lambda: defaultdict(dict))
    steps = 200

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            cfg.data.posterior_samples_num = sample_nums
            model = instantiate(cfg.model, data_config=cfg.data)
            start = time.perf_counter()
            fisher_estimator = PosteriorFDParametric(model=model)
            optimizer = OptimizationCornerPointsUnivariateGaussian(
                fisher_estimator,
                cfg.ksd.optimize.prior.Gaussian,
                cfg.ksd.optimize.loss.GaussianLogLikelihood
            )
            prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
            elapsed = time.perf_counter() - start
            largest_fd = prior_corners[0][2]
            times_list_parametric.append((sample_nums, elapsed))
            times_parametric[sample_nums][step] = elapsed
            print(f"***Parametric*** Samples: {sample_nums}, Initial FD: {largest_fd:.4f}, Time: {elapsed:.3f} sec")

    data_path = os.path.join(get_original_cwd(), "data/univariate_gaussian/runtimes/")
    os.makedirs(data_path, exist_ok=True)
    with open(data_path + "parametric_qcqp_optimisation_times.json", "w") as f:
        json.dump(times_parametric, f, indent=4)

    samples_nums_list = [int(x) for x in np.linspace(500, 10000, 10)]
    for step in range(steps):
        print(f"Non-parametric QCQP running step {step}.")
        for sample_nums in samples_nums_list:
            for basis_funcs_num in basis_funcs_num_list:
                cfg.data.posterior_samples_num = sample_nums
                cfg.data.prior_samples_num = sample_nums
                cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
                model = instantiate(cfg.model, data_config=cfg.data)
                start = time.perf_counter()
                estimator_prior = PriorFDNonParametric(model=model)
                estimator_posterior = PosteriorFDNonParametric(model=model)
                optimizer = OptimisationNonparametricBase(
                    estimator_posterior,
                    estimator_prior,
                    cfg.ksd.optimize.prior.nonparametric,
                    radius=2.0
                )
                result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
                elapsed = time.perf_counter() - start
                largest_fd = result_sdp_dual["dual_value"]
                times_list_nonparametric.append((sample_nums*2, basis_funcs_num, elapsed))
                times_nonparametric[sample_nums*2][basis_funcs_num][step] = elapsed
                print(
                    f"***Non-parametric*** Samples: {sample_nums*2}, Basis Functions num: {basis_funcs_num}, Initial FD: {largest_fd:.4f}, Time: {elapsed:.3f} sec")

    with open(data_path + "nonparametric_qcqp_optimisation_times.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian")
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
    steps = 200

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            cfg.data.posterior_samples_num = sample_nums
            model = instantiate(cfg.model, data_config=cfg.data)
            start = time.perf_counter()
            fisher_estimator = PosteriorFDParametric(model=model)
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
    with open(data_path + "parametric_qcqp_optimisation_times.json", "w") as f:
        json.dump(times_parametric, f, indent=4)

    samples_nums_list = [int(x) for x in np.linspace(500, 10000, 10)]
    for step in range(steps):
        print(f"Parametric running step {step}.")
        for sample_nums in samples_nums_list:
            for basis_funcs_num in basis_funcs_num_list:
                cfg.data.posterior_samples_num = sample_nums
                cfg.data.prior_samples_num = sample_nums
                cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
                model = instantiate(cfg.model, data_config=cfg.data)
                start = time.perf_counter()
                estimator_prior = PriorFDNonParametric(model=model)
                estimator_posterior = PosteriorFDNonParametric(model=model)
                optimizer = OptimisationNonparametricBase(
                    estimator_posterior,
                    estimator_prior,
                    cfg.ksd.optimize.prior.nonparametric,
                    radius=2.0
                )
                result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
                elapsed = time.perf_counter() - start
                largest_ksd = result_sdp_dual["dual_value"]
                times_list_nonparametric.append((sample_nums*2, basis_funcs_num, elapsed))
                times_nonparametric[sample_nums*2][basis_funcs_num][step] = elapsed
                print(
                    f"***Non-parametric*** Samples: {sample_nums*2}, Basis Functions num: {basis_funcs_num}, Initial FD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    with open(data_path + "nonparametric_qcqp_optimisation_times.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian")
def run_multivariate_gaussian_priors_diff_basis_funcs_num(cfg) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    times_list_parametric, times_list_nonparametric = [], []
    basis_funcs_num_list = [int(x) for x in np.linspace(5, 25, 21)]
    times_nonparametric = defaultdict(dict)
    steps = 100

    for step in range(steps):
        print(f"Parametric running step {step}.")
        for basis_funcs_num in basis_funcs_num_list:
            cfg.data.posterior_samples_num = 1000
            cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
            model = instantiate(cfg.model, data_config=cfg.data)
            estimator_prior = PriorFDNonParametric(model=model)
            estimator_posterior = PosteriorFDNonParametric(model=model)
            optimizer = OptimisationNonparametricBase(
                estimator_posterior,
                estimator_prior,
                cfg.ksd.optimize.prior.nonparametric,
                radius=2.0
            )
            start = time.perf_counter()
            result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
            elapsed = time.perf_counter() - start
            largest_ksd = result_sdp_dual["dual_value"]
            times_list_nonparametric.append((basis_funcs_num, elapsed))
            times_nonparametric[basis_funcs_num][step] = elapsed
            print(
                f"***Non-parametric*** Basis Functions num: {basis_funcs_num}, Initial FD: {largest_ksd:.4f}, Time: {elapsed:.3f} sec")

    data_path = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/")
    with open(data_path + "nonparametric_optimisation_times_diff_basis_funcs_nums.json", "w") as f:
        json.dump(times_nonparametric, f, indent=4)


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian")
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/toy/", config_name="multivariate_gaussian")
def run_priors_optimisation_runtimes(cfg):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    data_path = os.path.join(get_original_cwd(), "data/multivariate_gaussian/runtimes/")

    with open(data_path + "nonparametric_qcqp_optimisation_times.json", "r") as f:
        nonparametric_optimisation_times = json.load(f)
    with open(data_path + "parametric_qcqp_optimisation_times.json", "r") as f:
        parametric_optimisation_times = json.load(f)
    plot_runtime_parametric_nonparametric_with_ci(
        parametric_optimisation_times,
        nonparametric_optimisation_times,
        plot_cfg,
        output_dir,
        filename="runtime_parametric_nonparametric_qcqp_multivariate.pdf"
    )

    # with open(data_path + "nonparametric_qcqp_optimisation_times.json", "r") as f:
    #     nonparametric_optimisation_times = json.load(f)
    # plot_runtime_nonparametric_with_ci(
    #     nonparametric_optimisation_times,
    #     plot_cfg,
    #     output_dir,
    #     filename="runtime_parametric_nonparametric_qcqp_univariate.pdf"
    # )


if __name__ == "__main__":
    run_gaussian_priors_nonparametric()
    # run_gaussian_priors_nonparametric_diff_radii()
    # run_multivariate_gaussian_priors_nonparametric_diff_radii()
    # run_multivariate_gaussian_priors_nonparametric_basis_funcs_nums()
    # run_gaussian_priors_nonparametric_diff_samples_num()
    # run_multivariate_gaussian_priors_diff_samples_num()
    # run_multivariate_gaussian_priors_diff_basis_funcs_num()
    # run_priors_optimisation_runtimes()
