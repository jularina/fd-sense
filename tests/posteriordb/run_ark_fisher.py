import os
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import time

from src.discrepancies.posterior_fisher import PosteriorFDBase
from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.utils.files_operations import load_plot_config
from src.utils.files_operations import deepcopy_cfg
from src.plots.paper.posterior_db_paper_funcs import *
from src.utils.choosers import pick_optimizer
from src.optimization.corner_points_fisher import (
    OptimizationCornerPointsCompositePrior
)
from src.plots.paper.sbi_paper_funcs import *
from src.discrepancies.posterior_fisher import PosteriorFDNonParametric
from src.discrepancies.prior_fisher import PriorFDNonParametric
from src.optimization.nonparametric_fisher import OptimisationNonparametricBase


warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


def _filter_components(cfg: DictConfig, keep_names: List[str]) -> DictConfig:
    new_cfg = deepcopy_cfg(cfg)
    comps = new_cfg.ksd.optimize.prior.Composite.components
    new_cfg.ksd.optimize.prior.Composite.components = [c for c in comps if c.get("name") in keep_names]
    return new_cfg


def _eval_corners_with_cfg(estimator, cfg: DictConfig) -> Tuple:
    optimizer = OptimizationCornerPointsCompositePrior(
        estimator, cfg.ksd.optimize.prior.Composite, precomputed_qfs=True)
    qf_corners, worst_corner_dict = optimizer.evaluate_all_prior_corners()
    rows = []
    for corners in qf_corners:
        rows.append({
            "prior_corner": list(corners[0]) if not isinstance(corners[0], dict) else corners[0],
            "value": float(corners[2]),
        })
    return rows, worst_corner_dict


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ark_posteriordb")
def main(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    print("=== FD for PosteriorDB model ===")
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.posterior_samples_init
    fisher_estimator = PosteriorFDBase(samples=posterior_samples, model=model, candidate_type="loss")
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher():.4f}")
    cfg_all = _filter_components(cfg, keep_names=["alpha", "sigma", "beta1", "beta2", "beta3", "beta4", "beta5"])
    start = time.perf_counter()
    optimizer = OptimizationCornerPointsCompositePrior(fisher_estimator,
                                                       cfg_all.ksd.optimize.prior.Composite,
                                                       cfg.ksd.optimize.loss.GaussianARLogLikelihood,
                                                       )
    qf_corners, corner_largest_sens = optimizer.evaluate_all_prior_corners()
    elapsed = time.perf_counter() - start
    print(f"Time for optimisation of all parameters at once: {elapsed:.3f} sec")

    # Non-parametric
    # start = time.perf_counter()
    # model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    # posterior_samples = model.posterior_samples_init
    # prior_samples = model.posterior_samples_init
    # prior_samples = prior_samples[np.random.choice(10000, size=2000, replace=False)]
    # estimator_prior = PriorFDNonParametric(samples=prior_samples, model=model, candidate_type="prior")
    # estimator_posterior = PosteriorFDNonParametric(samples=posterior_samples, model=model, candidate_type="prior")
    # optimizer = OptimisationNonparametricBase(
    #     estimator_posterior,
    #     estimator_prior,
    #     cfg.ksd.optimize.prior.nonparametric,
    #     radius_lower_bound=100
    # )
    # result_sdp = optimizer.optimize_through_sdp_relaxation()
    # elapsed = time.perf_counter() - start
    # print(f"Time for optimisation of all parameters at once: {elapsed:.3f} sec")

    plot_three_panel_priors(
        rows_all=corner_largest_sens,
        cfg=cfg,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        sample_n_alpha=30,
        sample_n_sigma=30,
        sample_n_beta=30,
        seed=27,
        filename="ark_param_three_panel_priors.pdf"
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ark_posteriordb")
def compare_complexities(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_complexity_bar(
        cfg=cfg.ksd.optimize.prior.Composite,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        mcmc_run_time=80,
        fd_run_param=175,
        fd_run_nonparam=250,
        prefix=prefix,
        filename="ark_computational_cost.pdf",
        use_log10=True,
        include_nonparametric=True,
        show_nonparam_breakdown=True,
    )


if __name__ == "__main__":
    main()
    # compare_complexities()
