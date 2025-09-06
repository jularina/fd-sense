import glob
import os
from hydra.utils import instantiate, get_original_cwd
import hydra
import numpy as np

from src.discrepancies.posterior_ksd import PosteriorKSDParametric
from src.optimization.corner_points import OptimizationCornerPointsUnivariateGaussian
from src.distributions.laplace import Laplace


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="laplace")
def run_laplace_priors(cfg, save_samples=False) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    # Samples
    base_dir = os.path.join(get_original_cwd(), "data", "laplace")
    out_obs_dir = os.path.join(base_dir, "observations")
    out_post_dir = os.path.join(base_dir, "posterior_samples_normal_ref")

    if save_samples:
        rng_seed = 27
        n_samples = 5000
        np.random.seed(rng_seed)
        sigma_n2 = 2.0 / 3.0
        sigma_n = np.sqrt(sigma_n2)
        x_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
        os.makedirs(out_obs_dir, exist_ok=True)
        os.makedirs(out_post_dir, exist_ok=True)

        def _tag(x):
            return str(x).replace(".", "_")

        for x in x_values:
            obs_arr = np.array([x], dtype=float)
            np.save(os.path.join(out_obs_dir, f"obs_x_{_tag(x)}.npy"), obs_arr)
            mu_n = (2.0 / 3.0) * x
            samples = np.random.normal(loc=mu_n, scale=sigma_n, size=(n_samples, 1))
            np.save(os.path.join(out_post_dir, f"posterior_samples_x_{_tag(x)}.npy"), samples)

    # Instantiate the Bayesian model with runtime overrides
    obs_files = sorted(glob.glob(os.path.join(out_obs_dir, "obs_x_*.npy")))
    post_files = sorted(glob.glob(os.path.join(out_post_dir, "posterior_samples_x_*.npy")))
    assert len(obs_files) == len(post_files), "Mismatch between obs and posterior sample files."

    for obs_path, post_path in zip(obs_files, post_files):
        cfg.data.observations_path = obs_path
        cfg.data.posterior_samples_path = post_path
        model = instantiate(cfg.model, data_config=cfg.data)
        posterior_samples = model.posterior_samples_init
        kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
        ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
        obs_val = float(np.load(obs_path).item())
        print(f"x = {obs_val:.1f}")
        # print(f"x = {obs_val:.1f} | Initial KSD: {ksd_estimator.estimate_ksd():.4f}")
        optimizer = OptimizationCornerPointsUnivariateGaussian(ksd_estimator, cfg.ksd.optimize.prior.Laplace, distribution_cls=Laplace)
        qf_prior = optimizer.evaluate_all_prior_corners()


if __name__ == "__main__":
    run_laplace_priors()