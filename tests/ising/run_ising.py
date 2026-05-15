import warnings
import os
from collections import defaultdict
from pathlib import Path
import time
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from src.optimization.nonparametric import OptimizationNonparametricBase
from src.discrepancies.prior_ksd import PriorKSDNonParametric
from src.discrepancies.posterior_ksd import PosteriorKSDParametric, PosteriorKSDNonParametric
from src.utils.files_operations import load_plot_config
from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.optimization.corner_points import (
    OptimizationCornerPointsGamma
)
from src.utils.display import print_optimised_corners_values
from src.plots.paper.ising_model_paper_funcs import *

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ising_model")
def main(cfg: DictConfig) -> None:
    print("=== KSD for Ising model ===")
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.posterior_samples_init
    kernel: BaseKernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    ksd_est = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    ksd_value = float(ksd_est.estimate_ksd())
    print(f"[KSD] Posterior KSD (baseline prior): {ksd_value:.2f}")

    #Parametric prior optimisation
    optimizer = OptimizationCornerPointsGamma(ksd_est, cfg.ksd.optimize.prior.Gamma, cfg.ksd.optimize.loss.IsingLikelihoodGivenGrads)
    prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
    worst_corner_params = {}
    worst_corner_params["family"] = "Gamma"
    worst_corner_params["params"] = worst_corner
    prior_combinations = optimizer.evaluate_all_prior_combinations()
    print_optimised_corners_values(prior_corners)

    #Parametric lr optimisation
    lr_corners = optimizer.evaluate_all_lr_corners()
    lr_combinations = optimizer.evaluate_full_lr_grid()
    data_path = os.path.join(get_original_cwd(), "data/ising_model/")
    file_id = Path(cfg.data.observations_path).stem
    np.save(data_path + file_id + "_lr_optimisation.npy", np.array(lr_combinations))

    # Plots
    # plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    # output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    # plot_cfg = load_plot_config(plot_config_path)
    # plot_theta_prior(
    #     theta="theta",
    #     worst_corner=worst_corner_params,
    #     cfg=cfg,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     sample_n=20,
    #     seed=27,
    #     filename="ising_experiment_theta_size8_param_ref_cand_priors.pdf"
    # )
    # plot_lr_vs_ksd(
    #     lr_grid=lr_combinations,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     filename="ising_experiment_theta_size8_lr_vs_ksd.pdf",
    #     xlabel=r"learning rate",
    # )

    # Non-parametric
    # model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    # posterior_samples = model.posterior_samples_init
    # kernel: BaseKernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    # ksd_estimator = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel)
    # prior_samples = model.prior_samples_init
    # kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)
    # ksd_estimator_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)
    # psi_sdp_list, ksd_estimates_list = [], []
    # basis_funcs_num_list = [3, 5, 10, 15, 25]
    # radii_list = [0.01, 0.05, 0.1, 0.5]
    # times = []
    # nonparam_metrics = defaultdict(dict)
    #
    # for radius in radii_list:
    #     # for basis_funcs_num in basis_funcs_num_list:
    #     #     cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
    #     time_start = time.time()
    #     optimizer = OptimizationNonparametricBase(
    #         ksd_estimator,
    #         ksd_estimator_prior,
    #         cfg.ksd.optimize.prior.nonparametric,
    #         radius_lower_bound=radius
    #     )
    #     result_sdp = optimizer.optimize_through_sdp_relaxation(nuggets_to_obj=False)
    #     nonparam_metrics[radius] = result_sdp["ksd_est"]
    #     psi_sdp_list.append(result_sdp["psi_opt"])
    #     ksd_estimates_list.append(result_sdp["ksd_est"])
    #     print(f"Radius: {radius}, basis funcs num: {cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"]}, ksd: {result_sdp["ksd_est"]}.")
    #     time_end = time.time()
    #     time_passed = time_end - time_start
    #     times.append(time_passed)
    #     break
    #
    # print(f"Average time of non-parametric optimisation: {sum(times) / len(times)}.")

    # plot_sdp_densities_only(
    #     basis_function=optimizer.basis_function,
    #     psi_sdp_list=psi_sdp_list,
    #     radius_labels=radii_list,
    #     ksd_estimates=ksd_estimates_list,
    #     prior_distribution=model.prior_init,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     domain=(-3, 12),
    #     resolution=300,
    #     filename = "ising_model_size6_nonparametric_optimisation_basisnum25_centerskmeans_densities_various_radii.pdf"
    # )

    # plot_basis_colored_by_eigenvector(
    #     basis_function=optimizer.basis_function,
    #     eigenvector=optimizer.principle_eigenvector,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     domain=(-3, 12),
    #     resolution=300,
    #     filename="ising_model_size6_nonparametric_basis_colored_by_top_eigvec.pdf",
    # )


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="ising_model")
def create_combined_plots(cfg: DictConfig):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    data_path = os.path.join(get_original_cwd(), "data/ising_model/")
    lr_combinations_4 = np.load(data_path + "PseudoBayes_size=4_theta=5.0_dnum=1000_pnum=2000_data_lr_optimisation.npy")
    lr_combinations_6 = np.load(data_path + "PseudoBayes_size=6_theta=5.0_dnum=1000_pnum=2000_data_lr_optimisation.npy")
    lr_combinations_8 = np.load(data_path + "PseudoBayes_size=8_theta=5.0_dnum=1000_pnum=2000_data_lr_optimisation.npy")

    plot_lr_vs_ksd_multi(
        lr_grids=[lr_combinations_4, lr_combinations_6, lr_combinations_8],
        ds=[4,6,8],
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename="ising_experiment_theta_sizes4_6_8_lr_vs_ksd.pdf",
        xlabel=r"learning rate",
    )


if __name__ == "__main__":
    main()
    create_combined_plots()
