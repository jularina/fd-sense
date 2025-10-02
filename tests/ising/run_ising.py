import os
import warnings
from collections import defaultdict

import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import time

from src.discrepancies.posterior_ksd import PosteriorKSDParametric, PosteriorKSDNonParametric
from src.discrepancies.prior_ksd import PriorKSDNonParametric
from src.bayesian_model.base import BayesianModel
from src.kernels.base import BaseKernel
from src.utils.files_operations import load_plot_config
from src.utils.files_operations import get_outdir, save_json, save_csv, deepcopy_cfg
from src.optimization.nonparametric import OptimizationNonparametricBase
from src.utils.choosers import pick_optimizer
from src.optimization.corner_points import (
    OptimizationCornerPointsCompositePrior
)
from src.plots.paper.sbi_paper_funcs import *

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ising_model")
def main(cfg: DictConfig) -> None:
    print("=== KSD for Ising model ===")
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    posterior_samples = model.posterior_samples_init
    kernel: BaseKernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)

    ksd_est = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
    ksd_value = float(ksd_est.estimate_ksd())
    print(f"[KSD] Posterior KSD (baseline prior): {ksd_value:.2f}")




if __name__ == "__main__":
    main()
