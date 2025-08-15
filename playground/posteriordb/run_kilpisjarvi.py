import warnings
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.discrepancies.posterior_ksd import PosteriorKSDParametric

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/ksd_calculation/real/", config_name="kilpisjarvi_posteriordb")
def main(cfg: DictConfig) -> None:
    print("=== Parametric KSD (Kilpisjarvi from the posteriordb) ===")
    print("Config overrides:\n" + OmegaConf.to_yaml(cfg.get("playground", {})))

    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.posterior_samples_init
    kernel: BaseKernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    ksd_est = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    ksd_value = float(ksd_est.estimate_ksd())
    print(f"[KSD] Posterior KSD (baseline hyperprior): {ksd_value:.6f}")


if __name__ == "__main__":
    main()
