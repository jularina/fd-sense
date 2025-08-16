import os
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from src.discrepancies.posterior_ksd import PosteriorKSDParametric
from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.utils.files_operations import load_plot_config
from src.utils.files_operations import get_outdir, save_json, save_csv, deepcopy_cfg
from src.plots.paper.posterior_db_paper_funcs import *
from src.utils.choosers import pick_optimizer

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

def _eval_corners_with_cfg(ksd_est, cfg_like: DictConfig) -> List[Dict]:
    # Only the optimizer is recreated; model/kernel/KSD stay the same.
    optimizer = pick_optimizer(cfg_like, ksd_est)
    qf_corners = optimizer.evaluate_all_prior_corners()
    rows = []
    for corners in qf_corners:
        rows.append({
            "prior_corner": list(corners[0]) if not isinstance(corners[0], dict) else corners[0],
            "value": float(corners[2]),
        })
    return rows

@hydra.main(version_base="1.1", config_path="../../configs/ksd_calculation/real/", config_name="ark_posteriordb")
def main(cfg: DictConfig) -> None:
    print("=== Parametric KSD (ArK pooled from the posteriordb) ===")
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.posterior_samples_init
    kernel: BaseKernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    ksd_est = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    ksd_value = float(ksd_est.estimate_ksd())
    print(f"[KSD] Posterior KSD (baseline hyperprior): {ksd_value:.2f}")

    optimizer = pick_optimizer(cfg, ksd_est)
    qf_corners = optimizer.evaluate_all_prior_corners()
    rows = []
    for corners in qf_corners:
        rows.append({
            "prior_corner": list(corners[0]) if not isinstance(corners[0], dict) else corners[0],
            "value": float(corners[2]),
        })

    prefix = cfg.playground.get("output_prefix", "ark_param")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    if cfg.playground.get("save_json", True):
        save_json({"posterior_ksd": ksd_value, "corners": rows}, os.path.join(output_dir, f"{prefix}_corners.json"))
    if cfg.playground.get("save_csv", True):
        save_csv(rows, ["prior_corner", "value"], os.path.join(output_dir, f"{prefix}_corners.csv"))
    rows_sorted = sorted(rows, key=lambda r: r["value"])[:10]

    print("[Corners]:")
    for r in rows_sorted:
        print("  ", r)


@hydra.main(version_base="1.1", config_path="../../configs/ksd_calculation/real/", config_name="ark_posteriordb")
def main_for_paper(cfg: DictConfig) -> None:
    print("=== Parametric KSD (ArK pooled from the posteriordb) ===")
    rows = _run_one_optimization(cfg)
    model: BayesianModel = hydra.utils.instantiate(cfg.model, data_config=cfg.data)

    # Plots
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    # 1) Time series
    y = model.observations
    plot_ar_time_series(
        y=y,
        plot_cfg=plot_cfg, output_dir=output_dir,
        filename=f"{prefix}_ar_pred_ribbon.pdf"
    )


@hydra.main(version_base="1.1", config_path="../../configs/ksd_calculation/real/", config_name="ark_posteriordb")
def main_for_paper_several_variants(cfg: DictConfig) -> None:
    print("=== Parametric KSD (ArK pooled from the posteriordb) ===")

    # 0) One-time instantiate model → samples → kernel → KSD estimator
    model: BayesianModel = hydra.utils.instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.posterior_samples_init
    kernel: BaseKernel = hydra.utils.instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    ksd_est = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    ksd_value = float(ksd_est.estimate_ksd())
    print(f"[KSD] Posterior KSD (baseline hyperprior): {ksd_value:.3f}")

    # A) beta1 + beta3 (as given)
    cfg_A = _filter_components(cfg, keep_names=["beta1", "beta3"])
    rows_A = _eval_corners_with_cfg(ksd_est, cfg_A)

    # B) beta1 only
    cfg_B = _filter_components(cfg, keep_names=["beta1"])
    rows_B = _eval_corners_with_cfg(ksd_est, cfg_B)

    # C) beta3 only
    cfg_C = _filter_components(cfg, keep_names=["beta3"])
    rows_C = _eval_corners_with_cfg(ksd_est, cfg_C)

    # Plots
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    # plot_three_panel_priors(
    #     rows_A=rows_A,
    #     rows_B=rows_B,
    #     rows_C=rows_C,
    #     cfg=cfg,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     prefix=prefix,
    #     beta1_mu_log_range=cfg.ksd.optimize.prior.Composite.components[0].parameters_box_range.ranges.mu_log,
    #     beta1_sigma_log_range=cfg.ksd.optimize.prior.Composite.components[0].parameters_box_range.ranges.sigma_log,
    #     beta3_mu_range=cfg.ksd.optimize.prior.Composite.components[1].parameters_box_range.ranges.mu,
    #     beta3_sigma_range=cfg.ksd.optimize.prior.Composite.components[1].parameters_box_range.ranges.sigma,
    #     sample_n_beta1=30,
    #     sample_n_beta3=30,
    #     seed=27
    # )

    plot_three_panel_priors(
        rows_A=rows_A,
        rows_B=rows_B,
        rows_C=rows_C,
        cfg=cfg,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        beta1_mu_range=cfg.ksd.optimize.prior.Composite.components[0].parameters_box_range.ranges.mu,
        beta1_sigma_range=cfg.ksd.optimize.prior.Composite.components[0].parameters_box_range.ranges.sigma,
        beta3_mu_range=cfg.ksd.optimize.prior.Composite.components[1].parameters_box_range.ranges.mu,
        beta3_sigma_range=cfg.ksd.optimize.prior.Composite.components[1].parameters_box_range.ranges.sigma,
        sample_n_beta1=30,
        sample_n_beta3=30,
        seed=27
    )



if __name__ == "__main__":
    # main()
    # main_for_paper()
    main_for_paper_several_variants()
