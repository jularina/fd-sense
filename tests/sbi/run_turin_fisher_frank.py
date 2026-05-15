import os
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import time

from src.utils.files_operations import load_plot_config
from src.discrepancies.posterior_fisher import PosteriorFDBase
from src.plots.paper.toy_paper_fisher_funcs import plot_gaussian_copula_grid_pair
from src.optimization.corner_points_fisher import (
    OptimizationCornerPointsCompositePrior
)

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/paper/real/", config_name="sbi_nle_turin")
def main(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "sbi")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    print("=== FD for PosteriorDB model (Frank copula) ===")
    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    print(f"Initial Fisher for prior: {fisher_estimator.estimate_fisher_prior_only():.4f}")

    optimizer = OptimizationCornerPointsCompositePrior(
        fisher_estimator,
        cfg.fd.optimize.prior.Composite,
        cfg.fd.optimize.loss.GaussianLogLikelihoodWithGivenGrads,
    )

    print("Starting Frank copula black-box optimisation.")
    start = time.perf_counter()
    copula_res = optimizer.black_box_optimize_frank_copula(
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

    print(f"Frank Copula theta_sup: {copula_res.lambda_sup}")
    print(f"Frank Copula val_sup: {copula_res.val_sup}")
    print(f"Frank Copula theta_inf: {copula_res.lambda_inf}")
    print(f"Frank Copula val_inf: {copula_res.val_inf}")
    print(f"Frank Copula S_hat: {copula_res.S_hat}")
    print(f"Frank Copula nfev_sup: {copula_res.nfev_sup}")
    print(f"Frank Copula nfev_inf: {copula_res.nfev_inf}")
    print(f"Time for Frank copula optimisation: {elapsed:.3f} sec.")

    print("Starting Frank copula grid evaluation.")
    start = time.perf_counter()
    copula_grid_g0, theta_star_g0, val_star_g0 = optimizer.evaluate_frank_copula_grid_and_argmax(
        lambda_range=(-0.5, 0.5),
        n_grid=500,
        idx_g0=0,
        idx_nu=2,
    )
    copula_grid_T, theta_star_T, val_star_T = optimizer.evaluate_frank_copula_grid_and_argmax(
        lambda_range=(-0.5, 0.5),
        n_grid=500,
        idx_g0=1,
        idx_nu=2,
    )
    elapsed = time.perf_counter() - start
    print(f"Grid theta^star (g0): {theta_star_g0}, FD={val_star_g0}")
    print(f"Grid theta^star (T):  {theta_star_T}, FD={val_star_T}")
    print(f"Time for Frank copula grid evaluation: {elapsed:.3f} sec.")

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
        filename=f"{prefix}_frank_copula_fd_grid.pdf",
        xlabel=r"$\theta$",
        ylabel=r"$\hat{\rho}_m^{\mathrm{FD}}(\tilde{\Pi}^{\theta})$",
        logy=True,
        ylim=global_ylim,
    )


if __name__ == "__main__":
    main()
