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
from src.utils.files_operations import get_outdir, save_json, save_csv
warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../configs/paper/ksd_calculation/toy/", config_name="univariate_gaussian")
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

    # Instantiate model and kernels
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    prior_samples = model.prior_samples_init
    posterior_samples = model.posterior_samples_init
    kernel_post = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
    kernel_prior = instantiate(cfg.ksd.kernel, reference_data=prior_samples)

    ksd_post = PosteriorKSDNonParametric(samples=posterior_samples, model=model, kernel=kernel_post)
    ksd_prior = PriorKSDNonParametric(samples=prior_samples, model=model, kernel=kernel_prior)
    optimizer = OptimizationNonparametricBase(
        posterior_ksd=ksd_post,
        prior_ksd=ksd_prior,
        config=cfg.ksd.optimize.prior.nonparametric,
        radius_lower_bound=cfg.ksd.optimize.prior.nonparametric.radius_lower_bound,
    )
    result = optimizer.optimize_through_sdp_relaxation()

    # Summarize and save
    psi = result.get("psi_opt")
    summary = {
        "radius_lower_bound": cfg.ksd.optimize.prior.nonparametric.radius_lower_bound,
        "ksd_est": float(result.get("ksd_est", np.nan)),
    }
    outdir = get_outdir(cfg)
    prefix = cfg.playground.get("output_prefix", "np")

    if cfg.playground.get("save_json", True):
        save_json(summary, os.path.join(outdir, f"{prefix}_sdp_summary.json"))
    if cfg.playground.get("save_csv", True):
        save_csv([summary], ["radius_lower_bound", "ksd_est"], os.path.join(outdir, f"{prefix}_sdp_summary.csv"))

    if cfg.playground.get("save_psi", False) and psi is not None:
        np.save(os.path.join(outdir, f"{prefix}_psi_opt.npy"), psi)
        print("[INFO] Saved psi_opt to", os.path.join(outdir, f"{prefix}_psi_opt.npy"))

    print("[SDP] Summary:", summary)


if __name__ == "__main__":
    main()
