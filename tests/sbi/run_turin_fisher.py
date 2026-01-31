import os
import warnings
from typing import List
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
from collections import defaultdict
import time
import json

from src.bayesian_model.base import BayesianModel
from src.utils.files_operations import load_plot_config
from src.utils.files_operations import deepcopy_cfg
from src.utils.choosers import pick_optimizer
from src.optimization.nonparametric_fisher import OptimisationNonparametricBase
from src.plots.paper.sbi_paper_funcs import *
from src.discrepancies.posterior_fisher import PosteriorFDNonParametric
from src.discrepancies.prior_fisher import PriorFDNonParametric
from src.plots.paper.toy_paper_fisher_funcs import plot_runtime_nonparametric_with_ci

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
    # model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    # estimator_posterior = PosteriorFDNonParametric(model=model)
    # print("FD: ", estimator_posterior.estimate_fisher_prior_only())
    # estimator_prior = PriorFDNonParametric(model=model)
    # psi_sdp_list, fd_estimates_list = [], []
    # basis_funcs_num_list = [8, 12, 16, 20, 24]
    # radii_list = [10, 15, 20]
    # nonparam_metrics, nonparam_eta_stars = defaultdict(dict), defaultdict(dict)
    #
    # for radius in radii_list:
    #     for basis_funcs_num in basis_funcs_num_list:
    #         cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
    #         optimizer = OptimisationNonparametricBase(
    #             estimator_posterior,
    #             estimator_prior,
    #             cfg.ksd.optimize.prior.nonparametric,
    #             radius=radius
    #         )
    #         start = time.perf_counter()
    #         result_sdp = optimizer.optimize_through_sdp_relaxation()
    #         elapsed = time.perf_counter() - start
    #         print(f"SDP primal relaxation time: {elapsed}")
    #
    #         start = time.perf_counter()
    #         result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
    #         elapsed = time.perf_counter() - start
    #         print(f"SDP dual time: {elapsed}")
    #
    #         nonparam_metrics[basis_funcs_num][radius] = result_sdp["primal_value"]
    #         nonparam_eta_stars[basis_funcs_num][radius] = result_sdp["eta_star"]
    #         psi_sdp_list.append(result_sdp["eta_star"])
    #         fd_estimates_list.append(result_sdp["primal_value"])
    #         print(f"Radius: {radius}, basis funcs num: {basis_funcs_num}, objective estimate: {result_sdp["primal_value"]}.")

    # Plots
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    # plot_ksd_heatmap(data_dict=nonparam_metrics, plot_cfg=plot_cfg, output_dir=output_dir, log_scale=False)

    # Optimisation times
    # times_nonparametric = defaultdict(dict)
    # steps = 10
    # basis_funcs_num_list = [8, 16, 24, 32, 40, 48]
    #
    # for step in range(steps):
    #     print(f"************* Parametric running step {step} *************.")
    #     for basis_funcs_num in basis_funcs_num_list:
    #         print(f"K={basis_funcs_num}*.")
    #         cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"] = basis_funcs_num
    #         model = instantiate(cfg.model, data_config=cfg.data)
    #         estimator_prior = PriorFDNonParametric(model=model)
    #         estimator_posterior = PosteriorFDNonParametric(model=model)
    #         optimizer = OptimisationNonparametricBase(
    #             estimator_posterior,
    #             estimator_prior,
    #             cfg.ksd.optimize.prior.nonparametric,
    #             radius=10
    #         )
    #         start = time.perf_counter()
    #         result_sdp_dual = optimizer.optimize_dual_sdp_lambda_t()
    #         elapsed = time.perf_counter() - start
    #         times_nonparametric[basis_funcs_num][step] = elapsed
    #
    data_path = os.path.join(get_original_cwd(), "data/sbi/runtimes/")
    # with open(data_path + "nonparametric_optimisation_times_diff_basis_funcs_nums.json", "w") as f:
    #     json.dump(times_nonparametric, f, indent=4)

    with open(data_path + "nonparametric_optimisation_times_diff_basis_funcs_nums.json", "r") as f:
        nonparametric_optimisation_times = json.load(f)

    plot_runtime_nonparametric_with_ci(
        nonparametric_optimisation_times,
        plot_cfg,
        output_dir,
        filename="runtime_nonparametric_qcqp_sbi.pdf"
    )

if __name__ == "__main__":
    main()
