import os
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import time

from src.discrepancies.posterior_ksd import PosteriorKSDParametric, PosteriorKSDNonParametric
from src.discrepancies.prior_ksd import PriorKSDNonParametric
from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.utils.files_operations import load_plot_config
from src.utils.files_operations import get_outdir, save_json, save_csv, deepcopy_cfg
from src.optimization.nonparametric import OptimizationNonparametricBase
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

    # Parametric lr
    lr_corners = optimizer.evaluate_all_lr_corners()
    lr_grid = optimizer.evaluate_all_lr_grid()

    # Non-parametric
    ksd_estimator = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel)
    prior_samples = model.prior_samples_init
    kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)
    ksd_estimator_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)
    psi_sdp_list, ksd_estimates_list, radius_labels = [], [], []

    for radius_lower_bound in [1, 2, 3]:
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

    # Plots
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_turin_four_theta_priors(
        largest_sens=corner_largest_sens,
        cfg=cfg,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename="sbi_experiment_turin_prior_four_panel.pdf",
    )


if __name__ == "__main__":
    main()
