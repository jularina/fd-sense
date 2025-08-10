import os
import warnings
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.discrepancies.posterior_ksd import PosteriorKSDParametric
from src.utils.choosers import pick_optimizer
from  src.utils.files_operations import get_outdir, save_json, save_csv, load_numpy_array
warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def main(cfg: DictConfig) -> None:
    """
    Example:
      python scripts/ksd_parametric_cli.py \
        --config-path configs/paper/ksd_calculation/toy \
        --config-name univariate_gaussian \
        playground.posterior_path=/path/to/posterior.npy

    Optional overrides:
      playground.output_prefix=param
      playground.save_json=true playground.save_csv=true
    """
    print("=== Parametric KSD (corner points) ===")
    print("Config overrides:" + OmegaConf.to_yaml(cfg.get("playground", {})))

    posterior_path = cfg.playground.get("posterior_path")
    if not posterior_path:
        raise ValueError("Please provide playground.posterior_path=<path-to-npy>")

    posterior_samples = load_numpy_array(posterior_path)

    # Instantiate model & kernel
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    kernel: BaseKernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    # KSD
    ksd_est = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    ksd_value = float(ksd_est.estimate_ksd())
    print(f"[KSD] Posterior KSD (parametric): {ksd_value:.6f}")

    # Corner-points optimization
    optimizer = pick_optimizer(cfg, ksd_est)
    qf_corners = optimizer.evaluate_all_prior_corners()

    # Serialize results
    rows = []
    for key, val in qf_corners.items():
        # 'key' is a tuple of prior parameter values defining a corner
        rows.append({
            "prior_corner": list(key) if not isinstance(key, dict) else key,
            "value": float(val),
        })

    prefix = cfg.playground.get("output_prefix", "param")
    outdir = get_outdir(cfg)

    if cfg.playground.get("save_json", True):
        save_json({"posterior_ksd": ksd_value, "corners": rows}, os.path.join(outdir, f"{prefix}_corners.json"))
    if cfg.playground.get("save_csv", True):
        save_csv(rows, ["prior_corner", "value"], os.path.join(outdir, f"{prefix}_corners.csv"))

    # Nice console preview (top-10 by value asc)
    rows_sorted = sorted(rows, key=lambda r: r["value"])[:10]
    print("[Corners] Top-10:")
    for r in rows_sorted:
        print("  ", r)


if __name__ == "__main__":
    main()