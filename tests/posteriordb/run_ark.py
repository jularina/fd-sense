import os
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import time

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

@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ark_posteriordb")
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ark_posteriordb")
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


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ark_posteriordb")
def main_for_paper_several_variants(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    print("=== Parametric KSD (ArK pooled from the posteriordb) ===")

    # 0) One-time instantiate model → samples → kernel → KSD estimator
    # model: BayesianModel = hydra.utils.instantiate(cfg.model, data_config=cfg.data)
    # plot_ar_results(model.observations, model.posterior_samples_init, plot_cfg=plot_cfg, save=True, output_dir=output_dir,
    #                 prefix=prefix)
    # posterior_samples = model.posterior_samples_init
    # kernel: BaseKernel = hydra.utils.instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    # ksd_est = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    # start = time.perf_counter()
    # ksd_value = float(ksd_est.estimate_ksd())
    # elapsed = time.perf_counter() - start
    # print(f"[KSD] Posterior KSD (baseline hyperprior): {ksd_value:.3f}")
    # print(f"Time for one KSD evaluation: {elapsed:.3f} sec")

    # start = time.perf_counter()
    # cfg_all = _filter_components(cfg, keep_names=["alpha", "sigma", "beta1", "beta2", "beta3", "beta4", "beta5"])
    # rows_all, worst_corner_dict = _eval_corners_with_cfg(ksd_est, cfg_all)
    # elapsed = time.perf_counter() - start
    # print(f"Time for optimisation of all parameters at ONCE: {elapsed:.3f} sec")
    #
    # plot_three_panel_priors(
    #     rows_all=worst_corner_dict,
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

    # plot_complexity_bar(
    #     cfg = cfg.ksd.optimize.prior.Composite, plot_cfg=plot_cfg, output_dir=output_dir, m=10000, D=7, P=14, H_total=14,prefix=prefix,filename="ark_computational_cost.pdf",
    #     nuts_exponent= 5 / 4, use_log10= True,
    # )

    plot_complexity_bar(
        cfg=cfg.ksd.optimize.prior.Composite,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        m=10_000,
        D=7,
        P=14,
        H_total=14,
        prefix=prefix,
        filename="ark_computational_cost.pdf",
        nuts_exponent=5 / 4,
        use_log10=True,
        include_nonparametric=True,
        l=10_000,
        K=5,
        w=2.37,
        eps=1e-3,
        show_nonparam_breakdown=True,
    )

    # params = ["sigma", "alpha", "beta1", "beta2", "beta3", "beta4", "beta5"]
    # for i in range(1, 7):
    #     params_used = params[:i]
    #     print(f"Started working with {len(params_used)} parameters.")
    #     cfg_new = _filter_components(cfg, keep_names=params_used)
    #     start = time.perf_counter()
    #     rows_new = _eval_corners_with_cfg(ksd_est, cfg_new)
    #     elapsed = time.perf_counter() - start
    #     print(f"Time for optimisation of {len(params_used)} parameters: {elapsed:.3f} sec")
    #
    # param = "sigma"
    # start = time.perf_counter()
    # cfg_sigma = _filter_components(cfg, keep_names=[param])
    # rows_sigma = _eval_corners_with_cfg(ksd_est, cfg_sigma)
    # elapsed = time.perf_counter() - start
    # print(f"Time for optimisation of 1 parameter {param}: {elapsed:.3f} sec")
    #
    # param = "alpha"
    # start = time.perf_counter()
    # cfg_alpha = _filter_components(cfg, keep_names=[param])
    # rows_alpha, worst_corner_dict = _eval_corners_with_cfg(ksd_est, cfg_alpha)
    # elapsed = time.perf_counter() - start
    # print(f"Time for {param}: {elapsed:.3f} sec")
    #
    # param = "beta1"
    # start = time.perf_counter()
    # cfg_beta1 = _filter_components(cfg, keep_names=[param])
    # rows_beta1, worst_corner_dict = _eval_corners_with_cfg(ksd_est, cfg_beta1)
    # elapsed = time.perf_counter() - start
    # print(f"Time for {param}: {elapsed:.3f} sec")
    #
    # cfg_beta2 = _filter_components(cfg, keep_names=["beta2"])
    # rows_beta2, worst_corner_dict = _eval_corners_with_cfg(ksd_est, cfg_beta2)
    #
    # cfg_beta3 = _filter_components(cfg, keep_names=["beta3"])
    # rows_beta3, worst_corner_dict = _eval_corners_with_cfg(ksd_est, cfg_beta3)
    #
    # cfg_beta4 = _filter_components(cfg, keep_names=["beta4"])
    # rows_beta4, worst_corner_dict = _eval_corners_with_cfg(ksd_est, cfg_beta4)
    #
    # cfg_beta5 = _filter_components(cfg, keep_names=["beta5"])
    # rows_beta5, worst_corner_dict = _eval_corners_with_cfg(ksd_est, cfg_beta5)

    # A) beta1 + beta3 (as given)
    # cfg_A = _filter_components(cfg, keep_names=["beta1", "beta3", "sigma"])
    # rows_A = _eval_corners_with_cfg(ksd_est, cfg_A)




if __name__ == "__main__":
    # main()
    # main_for_paper()
    main_for_paper_several_variants()
