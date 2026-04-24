import itertools
import os
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import time

from src.utils.files_operations import load_plot_config
from src.discrepancies.posterior_fisher import PosteriorFDBase
from src.plots.paper.toy_paper_fisher_funcs import plot_gaussian_copula_grid_pair, plot_copula_all_pairs
from src.optimization.corner_points_fisher import (
    OptimizationCornerPointsCompositePrior
)

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")

PARAM_NAMES = {0: r"$G_0$", 1: r"$T$", 2: r"$\nu$", 3: r"$\sigma_{W^2}$"}


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="sbi_nle_turin")
def main(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "sbi")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    print("=== FD for PosteriorDB model (FGM copula) ===")
    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    print(f"Initial Fisher for prior: {fisher_estimator.estimate_fisher_prior_only():.4f}")

    optimizer = OptimizationCornerPointsCompositePrior(
        fisher_estimator,
        cfg.fd.optimize.prior.Composite,
        cfg.fd.optimize.loss.GaussianLogLikelihoodWithGivenGrads,
    )

    print("Starting FGM copula black-box optimisation.")
    start = time.perf_counter()
    copula_res = optimizer.black_box_optimize_fgm_copula(
        lambda_range=(-0.1, 0.1),
        seed=0,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        polish=True,
        workers=1,
        updating="deferred",
    )
    elapsed = time.perf_counter() - start

    print(f"FGM Copula lambda_sup: {copula_res.lambda_sup}")
    print(f"FGM Copula val_sup: {copula_res.val_sup}")
    print(f"FGM Copula lambda_inf: {copula_res.lambda_inf}")
    print(f"FGM Copula val_inf: {copula_res.val_inf}")
    print(f"FGM Copula S_hat: {copula_res.S_hat}")
    print(f"FGM Copula nfev_sup: {copula_res.nfev_sup}")
    print(f"FGM Copula nfev_inf: {copula_res.nfev_inf}")
    print(f"Time for FGM copula optimisation: {elapsed:.3f} sec.")

    print("Starting FGM copula grid evaluation (2-pair plot).")
    start = time.perf_counter()
    copula_grid_g0, lambda_star_g0, val_star_g0 = optimizer.evaluate_fgm_copula_grid_and_argmax(
        lambda_range=(-0.1, 0.1),
        n_grid=500,
        idx_g0=0,
        idx_nu=2,
    )
    copula_grid_T, lambda_star_T, val_star_T = optimizer.evaluate_fgm_copula_grid_and_argmax(
        lambda_range=(-0.1, 0.1),
        n_grid=500,
        idx_g0=1,
        idx_nu=2,
    )
    elapsed = time.perf_counter() - start
    print(f"Grid lambda^star (g0): {lambda_star_g0}, FD={val_star_g0}")
    print(f"Grid lambda^star (T):  {lambda_star_T}, FD={val_star_T}")
    print(f"Time for FGM copula grid evaluation: {elapsed:.3f} sec.")

    all_values = (
        [x[1] for x in copula_grid_g0]
        + [x[1] for x in copula_grid_T]
    )
    global_ylim = (min(all_values), max(all_values))

    plot_gaussian_copula_grid_pair(
        copula_grid_0=copula_grid_g0,
        copula_grid_1=copula_grid_T,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        filename=f"{prefix}_fgm_copula_fd_grid.pdf",
        xlabel=r"$\theta$",
        ylabel=r"$\hat{\rho}_m^{\mathrm{FD}}(\tilde{\Pi}^{\theta})$",
        logy=True,
        ylim=global_ylim,
    )

    print("Starting FGM copula grid evaluation (all pairs, range -0.5 to 0.5).")
    start = time.perf_counter()
    grids_and_labels = []
    for i, j in itertools.combinations(range(4), 2):
        grid, lam_star, val_star = optimizer.evaluate_fgm_copula_grid_and_argmax(
            lambda_range=(-0.5, 0.5),
            n_grid=500,
            idx_g0=i,
            idx_nu=j,
        )
        label = f"({PARAM_NAMES[i]}, {PARAM_NAMES[j]})"
        grids_and_labels.append((grid, label))
        print(f"Pair {label}: lambda^star={lam_star:.4f}, FD={val_star:.6f}")
    elapsed = time.perf_counter() - start
    print(f"Time for all-pairs grid evaluation: {elapsed:.3f} sec.")

    all_pair_values = [v for grid, _ in grids_and_labels for _, v in grid]
    plot_copula_all_pairs(
        grids_and_labels=grids_and_labels,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename=f"{prefix}_fgm_copula_fd_all_pairs.pdf",
        xlabel=r"$\lambda_c$",
        ylabel=r"$\hat{\rho}_m^{\mathrm{FD}}(\tilde{\Pi}^{\theta})$",
        logy=False,
        ylim=(min(all_pair_values), max(all_pair_values)),
    )


if __name__ == "__main__":
    main()
