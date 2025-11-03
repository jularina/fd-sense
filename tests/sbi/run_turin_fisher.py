import os
import warnings
from typing import List
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
from collections import defaultdict

from src.bayesian_model.base import BayesianModel
from src.utils.files_operations import load_plot_config
from src.utils.files_operations import deepcopy_cfg
from src.utils.choosers import pick_optimizer
from src.optimization.corner_points_fisher import (
    OptimizationCornerPointsCompositePrior,
)
from src.optimization.nonparametric_fisher import OptimisationNonparametricBase
from src.plots.paper.sbi_paper_funcs import *
from src.discrepancies.posterior_fisher import PosteriorFDBase, PosteriorFDNonParametric
from src.discrepancies.prior_fisher import PriorFDNonParametric


warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


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

    # Parametric prior
    # print("=== FD for SBI NLE Turin model ===")
    # model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    # posterior_samples = model.posterior_samples_init
    # fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="loss")
    # print(f"Initial Fisher: {fisher_estimator.estimate_fisher():.4f}")
    # optimizer = OptimizationCornerPointsCompositePrior(fisher_estimator,
    #                                                    cfg.ksd.optimize.prior.Composite,
    #                                                    cfg.ksd.optimize.loss.GaussianLogLikelihoodWithGivenGrads,
    #                                                    )
    # qf_corners, corner_largest_sens = optimizer.evaluate_all_prior_corners()
    # rows = []
    # for corners in qf_corners:
    #     rows.append({
    #         "prior_corner": list(corners[0]) if not isinstance(corners[0], dict) else corners[0],
    #         "value": float(corners[2]),
    #     })
    # rows_sorted = sorted(rows, key=lambda r: r["value"])[:10]
    # print("[Corners]:")
    # for r in rows_sorted:
    #     print("  ", r)
    # lr_corners = optimizer.evaluate_all_lr_corners()
    # lr_grid = optimizer.evaluate_all_lr_grid()
    # plot_turin_four_theta_priors(
    #     largest_sens=corner_largest_sens,
    #     cfg=cfg,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     filename="sbi_experiment_turin_prior_four_panel.pdf",
    # )

    # Non-parametric
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.posterior_samples_init
    prior_samples = model.prior_samples_init
    prior_samples = prior_samples[np.random.choice(10000, size=2000, replace=False)]
    estimator_prior = PriorFDNonParametric(samples=prior_samples, model=model, candidate_type="prior")
    estimator_posterior = PosteriorFDNonParametric(samples=posterior_samples, model=model, candidate_type="prior")
    psi_sdp_list, ksd_estimates_list = [], []
    basis_funcs_num_list = [10, 15, 25]
    radii_list = [5, 10, 50, 100]
    nonparam_metrics = defaultdict(dict)

    for radius in radii_list:
        for basis_funcs_num in basis_funcs_num_list:
            cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
            optimizer = OptimisationNonparametricBase(
                estimator_posterior,
                estimator_prior,
                cfg.ksd.optimize.prior.nonparametric,
                radius_lower_bound=radius
            )
            result_sdp = optimizer.optimize_through_sdp_relaxation()
            nonparam_metrics[basis_funcs_num][radius] = result_sdp["est"]
            psi_sdp_list.append(result_sdp["psi_opt"])
            ksd_estimates_list.append(result_sdp["est"])
            print(f"Radius: {radius}, basis funcs num: {basis_funcs_num}, objective estimate: {result_sdp["est"]}.")

    plot_ksd_heatmap(data_dict=nonparam_metrics, plot_cfg=plot_cfg, output_dir=output_dir, log_scale=False)


if __name__ == "__main__":
    main()
