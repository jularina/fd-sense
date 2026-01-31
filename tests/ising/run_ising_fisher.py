import warnings
import os
from collections import defaultdict
import time
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from src.utils.files_operations import load_plot_config
from src.plots.paper.toy_paper_fisher_funcs import plot_single_param
from src.plots.paper.ising_model_paper_funcs import *
from src.discrepancies.posterior_fisher import PosteriorFDBase, PosteriorFDNonParametric

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ising_model")
def main(cfg: DictConfig) -> None:
    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    print(f"Initial Fisher: {fisher_estimator.estimate_fisher_lr_only():.4f}")

    results = {}
    lr_range = cfg.fd.optimize.loss.IsingLikelihoodGivenGrads.parameters_box_range.ranges.lr
    for lr in np.linspace(lr_range[0], lr_range[1], 10):
        model.set_lr_parameter(lr)
        fisher_estimator = PosteriorFDBase(model=model)
        fisher = fisher_estimator.estimate_fisher_lr_only()
        results[lr] = fisher
        print(f"Lr: {lr}, FD: {fisher:.4f}")

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_single_param(results, "lr", plot_cfg, output_dir)


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ising_model")
def create_combined_plots(cfg: DictConfig):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    data_path = os.path.join(get_original_cwd(), "data/ising_model/fisher/")
    lr_combinations_4 = np.load(data_path + "PseudoBayes_size=4_theta=5.0_dnum=1000_pnum=2000_data_lr_optimisation.npy")
    lr_combinations_6 = np.load(data_path + "PseudoBayes_size=6_theta=5.0_dnum=1000_pnum=2000_data_lr_optimisation.npy")
    lr_combinations_8 = np.load(data_path + "PseudoBayes_size=8_theta=5.0_dnum=1000_pnum=2000_data_lr_optimisation.npy")

    plot_lr_vs_ksd_multi(
        lr_grids=[lr_combinations_4, lr_combinations_6, lr_combinations_8],
        ds=[4, 6, 8],
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filename="ising_experiment_theta_sizes4_6_8_lr_vs_ksd.pdf",
        xlabel=r"lr",
        ylbl="estimatedFDposteriorsShort",
    )


if __name__ == "__main__":
    main()
    # create_combined_plots()
