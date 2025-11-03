import warnings
import os
from collections import defaultdict
import time
from pathlib import Path
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from src.utils.files_operations import load_plot_config
from src.bayesian_model.base import BayesianModel
from src.discrepancies.posterior_fisher import PosteriorFDNonParametric, PosteriorFDBase
from src.discrepancies.prior_fisher import PriorFDNonParametric
from src.optimization.nonparametric_fisher import OptimisationNonparametricBase
from src.optimization.corner_points_fisher import OptimizationCornerPointsGamma
from src.plots.paper.ising_model_paper_funcs import *

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ising_model")
def main(cfg: DictConfig) -> None:
    print("=== FD for Ising model ===")
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.posterior_samples_init
    fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="loss")
    fisher_value = float(fisher_estimator.estimate_fisher())
    print(f"[FD] Posterior FD (baseline prior): {fisher_value:.2f}")

    # Parametric prior optimisation
    optimizer = OptimizationCornerPointsGamma(fisher_estimator, cfg.ksd.optimize.prior.Gamma, cfg.ksd.optimize.loss.IsingLikelihoodGivenGrads)
    prior_corners, worst_corner = optimizer.evaluate_all_prior_corners()
    worst_corner_params = {}
    worst_corner_params["family"] = "Gamma"
    worst_corner_params["params"] = worst_corner
    prior_combinations = optimizer.evaluate_all_prior_combinations()

    # Parametric lr optimisation
    lr_corners = optimizer.evaluate_all_lr_corners()
    lr_combinations = optimizer.evaluate_full_lr_grid()
    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")
    file_id = Path(cfg.data.observations_path).stem
    # np.save(data_path + file_id + "_lr_optimisation.npy", np.array(lr_combinations))

    # Plots
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_theta_prior(
        worst_corner=worst_corner_params,
        cfg=cfg,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        sample_n=20,
        seed=27,
        filename="ising_experiment_theta_size6_param_ref_cand_priors.pdf"
    )

    # Non-parametric
    # model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    # posterior_samples = model.posterior_samples_init
    # prior_samples = model.prior_samples_init
    # estimator_prior = PriorFDNonParametric(samples=prior_samples, model=model, candidate_type="prior")
    # estimator_posterior = PosteriorFDNonParametric(samples=posterior_samples, model=model, candidate_type="prior")
    #
    # psi_sdp_list, ksd_estimates_list = [], []
    # radii_list = [0.05, 0.1, 0.5, 10]
    # times = []
    # nonparam_metrics = defaultdict(dict)
    #
    # for radius in radii_list:
    #     time_start = time.time()
    #     optimizer = OptimisationNonparametricBase(
    #         estimator_posterior,
    #         estimator_prior,
    #         cfg.ksd.optimize.prior.nonparametric,
    #         radius_lower_bound=radius
    #     )
    #     result_sdp = optimizer.optimize_through_sdp_relaxation(nuggets_to_obj=False)
    #     nonparam_metrics[radius] = result_sdp["est"]
    #     psi_sdp_list.append(result_sdp["psi_opt"])
    #     ksd_estimates_list.append(result_sdp["est"])
    #     print(
    #         f"Radius: {radius}, basis funcs num: {cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"]}, obj: {result_sdp["est"]}.")
    #     time_end = time.time()
    #     time_passed = time_end - time_start
    #     times.append(time_passed)
    #
    # print(f"Average time of non-parametric optimisation: {sum(times) / len(times)}.")
    #
    # plot_sdp_densities_only(
    #     basis_function=optimizer.basis_function,
    #     psi_sdp_list=psi_sdp_list,
    #     radius_labels=radii_list,
    #     ksd_estimates=ksd_estimates_list,
    #     prior_distribution=model.prior_init,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     domain=(-3, 15),
    #     resolution=300,
    #     filename="ising_model_size6_nonparametric_optimisation_basisnum10_centersfarthest_densities_various_radii.pdf",
    #     ylbl="estimatedFDposteriorsShort"
    # )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ising_model")
def create_combined_plots(cfg: DictConfig):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")
    lr_combinations_4 = np.load(data_path + "PseudoBayes_size=4_theta=5.0_dnum=1000_pnum=2000_data_lr_optimisation.npy")
    lr_combinations_6 = np.load(data_path + "PseudoBayes_size=6_theta=5.0_dnum=1000_pnum=2000_data_lr_optimisation.npy")
    lr_combinations_8 = np.load(data_path + "PseudoBayes_size=8_theta=5.0_dnum=1000_pnum=2000_data_lr_optimisation.npy")

    plot_lr_vs_ksd_multi(
        lr_grids=[lr_combinations_4, lr_combinations_6, lr_combinations_8],
        ds=[4, 6, 8],
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename="ising_experiment_theta_sizes4_6_8_lr_vs_ksd.pdf",
        xlabel=r"lr",
        ylbl="estimatedFDposteriorsShort",
    )


if __name__ == "__main__":
    main()
    # create_combined_plots()
