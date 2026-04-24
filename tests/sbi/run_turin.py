import os
import warnings
from typing import List
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from src.discrepancies.posterior_ksd import PosteriorKSDParametric
from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.utils.files_operations import load_plot_config
from src.utils.files_operations import deepcopy_cfg
from src.utils.choosers import pick_optimizer
from src.optimization.corner_points import (
    OptimizationCornerPointsCompositePrior
)
from src.plots.paper.sbi_paper_funcs import *

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


def _run_one_optimization(cfg: DictConfig) -> List[Dict]:
    """Instantiate model/kernel, compute baseline KSD, evaluate corners, return rows."""
    model: BayesianModel = hydra.utils.instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.posterior_samples_init
    kernel: BaseKernel = hydra.utils.instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    ksd_est = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    ksd_value = float(ksd_est.estimate_ksd())
    print(f"[KSD] Posterior KSD (baseline hyperprior): {ksd_value:.3f}")

    optimizer = pick_optimizer(cfg, ksd_est)
    qf_corners = optimizer.evaluate_all_prior_corners()

    rows = []
    for corners in qf_corners:
        rows.append({
            "prior_corner": list(corners[0]) if not isinstance(corners[0], dict) else corners[0],
            "value": float(corners[2]),
        })
    return rows


def _filter_components(cfg: DictConfig, keep_names: List[str]) -> DictConfig:
    new_cfg = deepcopy_cfg(cfg)
    comps = new_cfg.ksd.optimize.prior.Composite.components
    new_cfg.ksd.optimize.prior.Composite.components = [c for c in comps if c.get("name") in keep_names]
    return new_cfg


def _eval_corners_with_cfg(ksd_est, cfg_like: DictConfig) -> Tuple:
    optimizer = pick_optimizer(cfg_like, ksd_est)
    qf_corners, worst_corner_dict = optimizer.evaluate_all_prior_corners()
    rows = []
    for corners in qf_corners:
        rows.append({
            "prior_corner": list(corners[0]) if not isinstance(corners[0], dict) else corners[0],
            "value": float(corners[2]),
        })
    return rows, worst_corner_dict


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="sbi_nle_turin")
def main(cfg: DictConfig) -> None:

    # Plots
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    print("=== KSD for SBI NLE Turin model ===")
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.posterior_samples_init
    kernel: BaseKernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    ksd_est = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    ksd_value = float(ksd_est.estimate_ksd())
    print(f"[KSD] Posterior KSD (baseline prior): {ksd_value:.2f}")

    # Parametric prior
    optimizer = OptimizationCornerPointsCompositePrior(ksd_est,
                                                       cfg.ksd.optimize.prior.Composite,
                                                       cfg.ksd.optimize.loss.GaussianLogLikelihoodWithGivenGrads,
                                                       precomputed_qfs=False
                                                       )
    qf_corners, corner_largest_sens = optimizer.evaluate_all_prior_corners()
    rows = []
    for corners in qf_corners:
        rows.append({
            "prior_corner": list(corners[0]) if not isinstance(corners[0], dict) else corners[0],
            "value": float(corners[2]),
        })
    rows_sorted = sorted(rows, key=lambda r: r["value"])[:10]
    print("[Corners]:")
    for r in rows_sorted:
        print("  ", r)

    lr_corners = optimizer.evaluate_all_lr_corners()
    lr_grid = optimizer.evaluate_all_lr_grid()

    # Non-parametric
    # model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    # posterior_samples = model.posterior_samples_init
    # kernel: BaseKernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    # ksd_estimator = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel)
    # prior_samples = model.prior_samples_init
    # kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)
    # ksd_estimator_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)
    # psi_sdp_list, ksd_estimates_list = [], []
    # basis_funcs_num_list = [2, 3]
    # radii_list = [0.1, 0.5, 5.0]
    basis_funcs_num_list = [3, 5, 10, 15, 25]
    radii_list = [0.05, 0.1, 0.5, 5.0]
    # nonparam_metrics = defaultdict(dict)
    #
    # for radius in radii_list:
    #     for basis_funcs_num in basis_funcs_num_list:
    #         # cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["degree"] = basis_funcs_num
    #         cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
    #         optimizer = OptimizationNonparametricBase(
    #             ksd_estimator,
    #             ksd_estimator_prior,
    #             cfg.ksd.optimize.prior.nonparametric,
    #             radius_lower_bound=radius
    #         )
    #         result_sdp = optimizer.optimize_through_sdp_relaxation(nuggets_to_obj=False)
    #         nonparam_metrics[basis_funcs_num][radius] = result_sdp["ksd_est"]
    #         psi_sdp_list.append(result_sdp["psi_opt"])
    #         ksd_estimates_list.append(result_sdp["ksd_est"])
    #         print(f"Radius: {radius}, basis funcs num: {basis_funcs_num}, ksd: {result_sdp["ksd_est"]}.")

#     nonparam_metrics = {2:
# {0.1: 0.059138362469974885, 0.5: 0.29254066591467137, 5.0: 2.908638509389252},
#
# 3:
# {0.1: 0.02398357319298377, 0.5: 0.11663343316146658, 5.0: 1.1481947690363985}}

    nonparam_metrics = {3:
                        {0.05: 0.43044202095702494, 0.1: 1.2812758050287645,
                            0.5: 2.8770761461387226, 5.0: 24.590955176277223},

                        5:
                        {0.05: 0.5046775802398636, 0.1: 1.0208909930170573,
                            0.5: 2.5295428951672685, 5.0: 26.305785468969624},

                        10:
                        {0.05: 1.9436202144940384, 0.1: 3.0387898898447534,
                            0.5: 10.024080659636736, 5.0: 110.32329413671725},

                        15:
                        {0.05: 2.131006102008526, 0.1: 3.5192233026478363, 0.5: 12.679214256103643, 5.0: 124.73881645552885},

                        25:
                        {0.05: 2.622211630337094, 0.1: 4.409561047561388, 0.5: 17.467573162214986, 5.0: 154.60061263245098}}

    # plot_ksd_heatmap(data_dict=nonparam_metrics, plot_cfg=plot_cfg, output_dir=output_dir)
    # plot_ksd_heatmap_continuous(
    #     data_dict=nonparam_metrics,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     colorbar_label=plot_cfg.plot.param_latex_names["logOptimisationProblem"],
    #     log_y=True,
    #     log_x=True,
    #     method="linear",
    # )
    # plot_turin_four_theta_priors(
    #     largest_sens=corner_largest_sens,
    #     cfg=cfg,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     filename="sbi_experiment_turin_prior_four_panel.pdf",
    # )
    plot_lr_vs_ksd(
        lr_grid=lr_grid,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename="sbi_experiment_turin_lr_vs_ksd.pdf",
        xlabel=r"learning rate",
    )


if __name__ == "__main__":
    main()
