import os
import warnings
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np

from src.bayesian_model.base import BayesianModel
from src.discrepancies.posterior_ksd import PosteriorKSDNonParametric
from src.discrepancies.prior_ksd import PriorKSDNonParametric
from src.optimization.nonparametric import OptimizationNonparametricBase
from  src.utils.files_operations import get_outdir, save_json, save_csv
warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
def main(cfg: DictConfig) -> None:
    """
    Example:
      python scripts/ksd_nonparametric_cli.py \
        --config-path configs/paper/ksd_calculation/toy \
        --config-name univariate_gaussian \
        playground.posterior_path=/path/to/post.npy \
        playground.prior_path=/path/to/prior.npy \
        playground.radius=3.0 \
        playground.save_psi=true

    Optional overrides:
      playground.output_prefix=np
      playground.save_json=true playground.save_csv=true
    """
    print("=== Nonparametric KSD (SDP) ===")
    print("Config overrides:" + OmegaConf.to_yaml(cfg.get("playground", {})))

    post_path = cfg.playground.get("posterior_path")
    prior_path = cfg.playground.get("prior_path")
    if not post_path or not prior_path:
        raise ValueError("Provide both playground.posterior_path and playground.prior_path (.npy files)")

    posterior_samples = np.load(post_path)
    prior_samples = np.load(prior_path)

    # Instantiate model and kernels
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)

    kernel_post = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)

    ksd_post = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel_post)
    ksd_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)

    radius = float(cfg.playground.get("radius", 3.0))

    optimizer = OptimizationNonparametricBase(
        ksd_estimator=ksd_post,
        ksd_estimator_prior=ksd_prior,
        optimize_cfg=cfg.ksd.optimize.prior.nonparametric,
        radius_lower_bound=radius,
    )

    result = optimizer.optimize_through_sdp_relaxation()

    # Summarize and save
    psi = result.get("psi_opt")
    psi_norm2 = float(psi.T @ psi) if psi is not None else None
    summary = {
        "radius_lower_bound": radius,
        "ksd_est": float(result.get("ksd_est", np.nan)),
        "psi_norm2": psi_norm2,
    }

    outdir = get_outdir(cfg)
    prefix = cfg.playground.get("output_prefix", "np")

    if cfg.playground.get("save_json", True):
        save_json(summary, os.path.join(outdir, f"{prefix}_sdp_summary.json"))
    if cfg.playground.get("save_csv", True):
        save_csv([summary], ["radius_lower_bound", "ksd_est", "psi_norm2"], os.path.join(outdir, f"{prefix}_sdp_summary.csv"))

    if cfg.playground.get("save_psi", False) and psi is not None:
        np.save(os.path.join(outdir, f"{prefix}_psi_opt.npy"), psi)
        print("[INFO] Saved psi_opt to", os.path.join(outdir, f"{prefix}_psi_opt.npy"))

    print("[SDP] Summary:", summary)


if __name__ == "__main__":
    main()