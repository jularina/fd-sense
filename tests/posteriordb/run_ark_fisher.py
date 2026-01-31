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

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class BlockDecomposition:
    name: str
    main_quad: float          # eta_j^T A_jj eta_j
    linear: float             # b_j^T eta_j
    interaction: float        # sum_{k!=j} eta_j^T A_jk eta_k   (counted ONCE per block; totals will be 2x this)
    total_with_half_interactions: float  # main + linear + interaction/?? (see note below)


def decompose_prior_qf_by_blocks(
    eta: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    c: float,
    names: List[str],
    block_size: int = 2,
) -> Tuple[Dict[str, BlockDecomposition], Dict[str, float]]:
    """
    Decompose Q(eta)=eta^T A eta + b^T eta + c into block contributions.

    Returns:
      per_block: dict of BlockDecomposition
      totals: dict with sanity-check totals
    """
    eta = np.asarray(eta).reshape(-1)
    b = np.asarray(b).reshape(-1)
    A = np.asarray(A)

    d = eta.shape[0]
    assert A.shape == (d, d), f"A shape {A.shape} does not match eta length {d}"
    assert b.shape == (d,), f"b length {b.shape} does not match eta length {d}"
    assert d == block_size * len(names), f"eta length {d} != {block_size} * len(names) {len(names)}"

    # Full value
    Q_full = float(eta @ A @ eta + b @ eta + c)
    per_block: Dict[str, BlockDecomposition] = {}

    # Precompute block indices
    block_inds = []
    for j, nm in enumerate(names):
        idx = np.arange(j * block_size, (j + 1) * block_size)
        block_inds.append((nm, idx))

    # Compute within-block and interaction pieces
    for j, (nm, idx_j) in enumerate(block_inds):
        eta_j = eta[idx_j]
        b_j = b[idx_j]
        A_jj = A[np.ix_(idx_j, idx_j)]

        main_quad = float(eta_j @ A_jj @ eta_j)
        linear = float(b_j @ eta_j)

        # interaction_j := sum_{k != j} eta_j^T A_jk eta_k  (no factor 2 here)
        interaction = 0.0
        for k, (_, idx_k) in enumerate(block_inds):
            if k == j:
                continue
            A_jk = A[np.ix_(idx_j, idx_k)]
            eta_k = eta[idx_k]
            interaction += float(eta_j @ A_jk @ eta_k)

        # A clean “per-block total” that sums back to Q (up to constant) is:
        # main + linear + 0.5 * (2 * sum_{j<k} eta_j^T A_jk eta_k) attributed evenly.
        # Since interaction here counts both directions across j, we use 0.5 * interaction.
        total_with_half_interactions = main_quad + linear + 0.5 * interaction

        per_block[nm] = BlockDecomposition(
            name=nm,
            main_quad=main_quad,
            linear=linear,
            interaction=interaction,
            total_with_half_interactions=total_with_half_interactions,
        )

    # Sanity checks
    main_sum = sum(v.main_quad for v in per_block.values())
    linear_sum = sum(v.linear for v in per_block.values())

    # Each cross term eta_j^T A_jk eta_k is counted twice in sum_j interaction_j (once as j→k and once as k→j),
    # so the true total cross contribution to eta^T A eta is:
    # cross_total = 0.5 * sum_j interaction_j
    cross_total = 0.5 * sum(v.interaction for v in per_block.values())

    Q_reconstructed_no_c = main_sum + cross_total + linear_sum
    Q_reconstructed = Q_reconstructed_no_c + float(c)

    totals = {
        "Q_full": Q_full,
        "main_sum": float(main_sum),
        "linear_sum": float(linear_sum),
        "cross_total": float(cross_total),
        "c": float(c),
        "Q_reconstructed": float(Q_reconstructed),
        "abs_err": float(abs(Q_reconstructed - Q_full)),
    }

    return per_block, totals


def rank_blocks(per_block: Dict[str, BlockDecomposition], key: str = "total_with_half_interactions"):
    """Return a list of (name, value) sorted descending by chosen key."""
    pairs = [(nm, getattr(obj, key)) for nm, obj in per_block.items()]
    return sorted(pairs, key=lambda x: x[1], reverse=True)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ark_posteriordb")
def main(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    print("=== FD for PosteriorDB model ===")
    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    print(f"Initial Fisher for prior: {fisher_estimator.estimate_fisher_prior_only():.4f}")
    print(f"Initial Fisher for lr: {fisher_estimator.estimate_fisher_lr_only():.4f}")

    start = time.perf_counter()
    optimizer = OptimizationCornerPointsCompositePrior(fisher_estimator,
                                                       cfg.fd.optimize.prior.Composite,
                                                       cfg.fd.optimize.loss.GaussianARLogLikelihood,
                                                       )
    qf_corners, eta_star = optimizer.evaluate_all_prior_corners()
    elapsed = time.perf_counter() - start
    print(f"Time for optimisation of all parameters at once: {elapsed:.3f} sec.")


    # Interpretation
    A, b, c = fisher_estimator.compute_fisher_quadratic_form_prior_only()
    names = ["alpha", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma"]
    per_block, totals = decompose_prior_qf_by_blocks(
        eta=eta_star, A=A, b=b, c=c, names=names, block_size=2
    )
    print(totals)
    ranking = rank_blocks(per_block, key="total_with_half_interactions")
    print("Ranked (main + linear + 0.5*interactions):")
    for nm, val in ranking:
        print(nm, val)

    # plot_three_panel_priors(
    #     rows_all=corner_largest_sens,
    #     cfg=cfg,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     prefix=prefix,
    #     sample_n_alpha=30,
    #     sample_n_sigma=30,
    #     sample_n_beta=30,
    #     seed=27,
    #     filename="ark_param_three_panel_priors.pdf"
    # )

    alpha_ref = {"family": "Gaussian", "params": {"mu": 0.0, "sigma": 10.0}}
    betas_ref = {"family": "Gaussian", "params": {"mu": 0.0, "sigma": 10.0}}
    sigma_ref = {"family": "HalfCauchy", "params": {"gamma": 2.5}}

    alpha_ms = {"family": "Gaussian", "params": {"mu": 20.0, "sigma": 10.0 / 3.0}}
    betas_ms = {
        "beta1": {"family": "Gaussian", "params": {"mu": -20.0, "sigma": 10.0 / 3.0}},
        "beta2": {"family": "Gaussian", "params": {"mu": -20.0, "sigma": 10.0 / 3.0}},
        "beta3": {"family": "Gaussian", "params": {"mu": -20.0, "sigma": 10.0 / 3.0}},
        "beta4": {"family": "Gaussian", "params": {"mu": 20.0, "sigma": 10.0 / 3.0}},
        "beta5": {"family": "Gaussian", "params": {"mu": 20.0, "sigma": 10.0 / 3.0}},
    }
    sigma_ms = {"family": "Gamma", "params": {"alpha": 4.0, "theta": 5.0 / 24.0}}
    alpha_box_ranges = {
        "mu": (-20.0, 20.0),
        "sigma": (10.0 / 3.0, 30.0),
    }
    beta_box_ranges = {
        "mu": (-20.0, 20.0),
        "sigma": (10.0 / 3.0, 30.0),
    }
    betas_box_ranges = {
        "beta1": beta_box_ranges,
        "beta2": beta_box_ranges,
        "beta3": beta_box_ranges,
        "beta4": beta_box_ranges,
        "beta5": beta_box_ranges,
    }
    sigma_box_ranges = {
        "alpha": (4.0 / 9.0, 4.0),
        "theta": (5.0 / 24.0, 135.0 / 8.0),
    }
    plot_three_panel_priors_all_betas_one_plot_explicit(
        alpha_ref=alpha_ref,
        betas_ref=betas_ref,
        sigma_ref=sigma_ref,
        alpha_ms=alpha_ms,
        betas_ms=betas_ms,
        sigma_ms=sigma_ms,
        alpha_box_ranges=alpha_box_ranges,
        betas_box_ranges=betas_box_ranges,
        sigma_box_ranges=sigma_box_ranges,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        sample_n_alpha=30,
        sample_n_sigma=30,
        sample_n_beta_total=150,
        seed=27,
        filename="ark_param_three_panel_priors.pdf",
    )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ark_posteriordb")
def compare_complexities(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    plot_complexity_bar(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        mcmc_run_time=80,
        posterior_evaluations_num=2**14,
        fd_run_param=81,
        fd_run_nonparam=0,
        prefix=prefix,
        filename="ark_computational_cost.pdf",
        use_log10=True,
        include_nonparametric=False,
        show_nonparam_breakdown=False,
    )


if __name__ == "__main__":
    # main()
    compare_complexities()
