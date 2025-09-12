from hydra.utils import instantiate, get_original_cwd
import hydra
from omegaconf import DictConfig
import os, glob, numpy as np
import pymc as pm

from src.discrepancies.posterior_ksd import PosteriorKSDParametric
from src.distributions.laplace import Laplace
from src.distributions.gaussian import Gaussian
from src.plots.paper.comparison import *
from src.utils.files_operations import load_plot_config
from src.optimization.corner_points import (
    OptimizationCornerPointsUnivariateGaussian,
    OptimizationCornerPointsCompositePrior
)


def sample_trunc_normal_loc1(x, a, b, n, rng):
    out = np.empty((n, 1), dtype=float)
    filled = 0
    # batch a bit for speed
    batch = max(1000, n // 2)
    while filled < n:
        z = rng.normal(loc=x, scale=1.0, size=(batch,))
        keep = (z >= a) & (z <= b)
        k = int(keep.sum())
        if k:
            take = min(k, n - filled)
            out[filled:filled+take, 0] = z[keep][:take]
            filled += take
    return out

@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="laplace_cauchy_for_comparison")
def run_laplace_cauchy_priors_uniform_reference(cfg: DictConfig, save_samples: bool = False) -> None:
    """
    Same as the 'normal reference' runner, but creates reference posterior samples
    assuming a Uniform(a,b) prior. With one observation x and N(x|theta,1) likelihood,
    the posterior is N(x,1) truncated to [a,b].
    """
    base_dir    = os.path.join(get_original_cwd(), "data", "uniform_reference")
    out_obs_dir = os.path.join(base_dir, "observations")
    out_post_dir = os.path.join(base_dir, "posterior_samples_uniform_ref")

    if save_samples:
        rng_seed  = 27
        n_samples = 5000
        rng = np.random.default_rng(rng_seed)
        a, b = -5.0, 5.0

        x_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
        os.makedirs(out_obs_dir, exist_ok=True)
        os.makedirs(out_post_dir, exist_ok=True)

        def _tag(x):
            return str(x).replace(".", "_")

        for x in x_values:
            obs_arr = np.array([x], dtype=float)
            np.save(os.path.join(out_obs_dir, f"obs_x_{_tag(x)}.npy"), obs_arr)
            samples = sample_trunc_normal_loc1(x=float(x), a=a, b=b, n=n_samples, rng=rng)
            np.save(os.path.join(out_post_dir, f"posterior_samples_x_{_tag(x)}.npy"), samples)

    obs_files  = sorted(glob.glob(os.path.join(out_obs_dir,  "obs_x_*.npy")))
    post_files = sorted(glob.glob(os.path.join(out_post_dir, "posterior_samples_x_*.npy")))
    assert len(obs_files) == len(post_files), "Mismatch between obs and posterior sample files."

    for obs_path, post_path in zip(obs_files, post_files):
        cfg.data.observations_path = obs_path
        cfg.data.posterior_samples_path = post_path
        model = instantiate(cfg.model, data_config=cfg.data)
        posterior_samples = model.posterior_samples_init
        kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
        ksd_estimator = PosteriorKSDParametric(samples=posterior_samples, model=model, kernel=kernel)
        x_val = float(np.load(obs_path).item())
        print(f"x = {x_val:.1f} | Initial KSD (Uniform ref): {ksd_estimator.estimate_ksd():.4f}")
        # optimizer = OptimizationCornerPointsUnivariateGaussian(ksd_estimator, cfg.ksd.optimize.prior.Laplace, distribution_cls=Laplace)
        # qf_prior = optimizer.evaluate_all_prior_corners()

@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="laplace_cauchy_for_comparison")
def run_laplace_cauchy_priors_normal_reference(cfg, save_samples=False) -> None:
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

        # Prior: θ ~ N(mu0, sigma0^2)
        mu0 = 0.0  # <-- set your prior mean
        sigma0 = 1.414  # <-- set your prior std

        # Likelihood: x | θ ~ N(θ, sigma_noise^2)
        sigma_noise2 = 1.0
        sigma_noise = np.sqrt(sigma_noise2)

        x_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
        os.makedirs(out_obs_dir, exist_ok=True)
        os.makedirs(out_post_dir, exist_ok=True)

        def _tag(x):
            return str(x).replace(".", "_")

        for x in x_values:
            obs_arr = np.array([x], dtype=float)
            np.save(os.path.join(out_obs_dir, f"obs_x_{_tag(x)}.npy"), obs_arr)
            tau0 = 1.0 / (sigma0 ** 2)
            tau_noise = 1.0 / sigma_noise2
            tau_post = tau0 + tau_noise

            mu_post = (tau0 * mu0 + tau_noise * x) / tau_post
            sigma_post2 = 1.0 / tau_post
            sigma_post = np.sqrt(sigma_post2)
            samples = np.random.normal(loc=mu_post, scale=sigma_post, size=(n_samples, 1))
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
        # print(f"x = {obs_val:.1f} | Initial KSD: {ksd_estimator.estimate_ksd():.4f}")
        optimizer = OptimizationCornerPointsUnivariateGaussian(ksd_estimator, cfg.ksd.optimize.prior.Laplace, distribution_cls=Laplace)
        qf_prior = optimizer.evaluate_all_prior_corners()


def _pymc_posterior_samples_ref(x: float,
                                prior_kind: str,
                                prior_params: dict,
                                draws: int,
                                tune: int,
                                target_accept: float,
                                random_seed: int) -> np.ndarray:
    """
    Sample p(theta | x) ∝ N(x | theta, 1) * prior(theta) using PyMC (NUTS).
    Supports: 'laplace', 'cauchy', 'uniform'.
    Returns (draws, 1) array of theta samples.
    """
    prior_kind = prior_kind.lower()
    with pm.Model() as m:
        if prior_kind == "laplace":
            mu = float(prior_params.get("mu"))
            b = float(prior_params.get("b"))
            theta = pm.Laplace("theta", mu=mu, b=b)
        elif prior_kind == "cauchy":
            x  = float(prior_params.get("x"))
            gamma = float(prior_params.get("gamma"))
            theta = pm.Cauchy("theta", alpha=x, beta=gamma)
        elif prior_kind == "uniform":
            a = float(prior_params.get("a"))
            b = float(prior_params.get("b"))
            theta = pm.Uniform("theta", lower=a, upper=b)
        else:
            raise ValueError("prior_kind must be one of {'laplace', 'cauchy', 'uniform'}")

        pm.Normal("x_obs", mu=theta, sigma=1.0, observed=float(x))

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=1,
            cores=1,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
        )

    samples = idata.posterior["theta"].values.reshape(-1, 1)
    return samples


def _make_tag(prior_kind: str, prior_params: dict) -> str:
    pk = prior_kind.lower()
    if pk == "laplace":
        return f"laplace_mu{prior_params.get('mu',0.0)}_b{prior_params.get('b',1.0)}"
    if pk == "cauchy":
        return f"cauchy_mu{prior_params.get('mu',0.0)}_g{prior_params.get('gamma',1.0)}"
    if pk == "uniform":
        return f"uniform_{prior_params.get('a',-5.0)}_{prior_params.get('b',5.0)}"
    return "unknown_ref"


@hydra.main(version_base="1.1",
            config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="laplace_cauchy_for_comparison")
def run_nonconjugate_reference_prior_mcmc(cfg: DictConfig, save_samples: bool = True) -> None:
    """
    1) (Optional) Sample and save reference posterior draws using a non-conjugate prior
       (Laplace/Cauchy/Uniform) with PyMC (NUTS).
    2) Loop over saved files and run the standard KSD pipeline (model → kernel → estimator).
    """
    base_dir     = os.path.join(get_original_cwd(), "data", "nonconj_ref_pymc")
    out_obs_dir  = os.path.join(base_dir, "observations")
    x_values = list(getattr(cfg.data, "x_values", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]))
    prior_kind   = 'cauchy'          # 'laplace' | 'cauchy' | 'uniform'
    prior_params = {'x': 0, 'gamma': 4}

    draws          = int(getattr(cfg.data, "ref_draws", 5000))
    tune           = int(getattr(cfg.data, "ref_tune", 2000))
    target_accept  = float(getattr(cfg.data, "ref_target_accept", 0.95))
    seed           = int(getattr(cfg.data, "ref_seed", 27))

    tag = _make_tag(prior_kind, prior_params)
    out_post_dir = os.path.join(base_dir, f"posterior_samples_{tag}")

    if save_samples:
        os.makedirs(out_obs_dir, exist_ok=True)
        os.makedirs(out_post_dir, exist_ok=True)

        def _tagx(x): return str(x).replace(".", "_")

        for x in x_values:
            np.save(os.path.join(out_obs_dir, f"obs_x_{_tagx(x)}.npy"),
                    np.array([x], dtype=float))
            samples = _pymc_posterior_samples_ref(
                x=float(x),
                prior_kind=prior_kind,
                prior_params=prior_params,
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                random_seed=seed,
            )
            np.save(os.path.join(out_post_dir, f"posterior_samples_x_{_tagx(x)}.npy"),
                    samples)

    obs_files  = sorted(glob.glob(os.path.join(out_obs_dir,  "obs_x_*.npy")))
    post_files = sorted(glob.glob(os.path.join(out_post_dir, "posterior_samples_x_*.npy")))
    assert len(obs_files) == len(post_files), "Mismatch between obs and posterior sample files."

    for obs_path, post_path in zip(obs_files, post_files):
        cfg.data.observations_path = obs_path
        cfg.data.posterior_samples_path = post_path
        model = instantiate(cfg.model, data_config=cfg.data)
        posterior_samples = model.posterior_samples_init
        kernel = instantiate(cfg.ksd.kernel, reference_data=posterior_samples)
        ksd_estimator = PosteriorKSDParametric(
            samples=posterior_samples,
            model=model,
            kernel=kernel
        )
        x_val = float(np.load(obs_path).item())
        print(f"[{prior_kind} ref] x = {x_val:.1f} | Initial KSD: {ksd_estimator.estimate_ksd():.4f}")


@hydra.main(version_base="1.1",
            config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="laplace_cauchy_for_comparison")
def plot_comparison_plots(cfg: DictConfig):
   plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
   plot_cfg = load_plot_config(plot_config_path)
   output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)

   # 1) Reference-based shape plots
   plot_reference_shapes_x(10.0, "normal", plot_cfg, output_dir)
   plot_reference_shapes_x(10.0, "uniform", plot_cfg, output_dir)
   plot_reference_shapes_x(10.0, "cauchy", plot_cfg, output_dir)

   # 2) Informal approach plot
   means = {
       r"$E[\theta|x]$ (Normal prior): 6.67": 6.67,
       r"$E[\theta|x]$ (Laplace prior): 9.27": 9.27,
       r"$E[\theta|x]$ (Cauchy prior): 9.80": 9.80,
   }
   plot_informal_means_x(10.0, means, plot_cfg, output_dir)


@hydra.main(version_base="1.1",
            config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="normals_for_comparison")
def run_normals_comparison(cfg: DictConfig, save_samples: bool = False):
    base_dir = os.path.join(get_original_cwd(), "data", "bivar_gaussian")
    out_obs_dir = os.path.join(base_dir, "observations")
    out_post_dir = os.path.join(base_dir, "posterior_samples_normal_ref")
    os.makedirs(out_obs_dir, exist_ok=True)
    os.makedirs(out_post_dir, exist_ok=True)

    if save_samples:
        rng_seed = 27
        n_samples = 5000
        np.random.seed(rng_seed)

        mu0 = np.array([0.0, 0.0])
        Sigma0 = np.eye(2)
        Sigma_noise = np.eye(2)
        x_list = [
            (0.0, 0.0),
            (0.0, 3.0),
            (3.0, 0.0),
        ]
        Sigma_post = 0.5 * np.eye(2)

        def _tag2(x1, x2):
            t1 = str(x1).replace(".", "_")
            t2 = str(x2).replace(".", "_")
            return f"{t1}_{t2}"

        for (x1, x2) in x_list:
            x = np.array([x1, x2], dtype=float)
            np.save(os.path.join(out_obs_dir, f"obs_x_{_tag2(x1, x2)}.npy"), x.reshape((-1,2)))
            mu_post = 0.5 * x
            samples = np.random.multivariate_normal(mean=mu_post, cov=Sigma_post, size=n_samples)
            np.save(os.path.join(out_post_dir, f"posterior_samples_x_{_tag2(x1, x2)}.npy"), samples)

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
        obs_val = np.load(obs_path)
        print(f"(x_0, x_1) = {obs_val} | Initial KSD: {ksd_estimator.estimate_ksd():.4f}")
        optimizer = OptimizationCornerPointsCompositePrior(ksd_estimator, cfg.ksd.optimize.prior.Composite, precomputed_qfs=False)
        qf_prior = optimizer.evaluate_all_prior_corners()


if __name__ == "__main__":
    # run_laplace_cauchy_priors_normal_reference()
    # run_laplace_cauchy_priors_uniform_reference()
    # run_nonconjugate_reference_prior_mcmc()
    # plot_comparison_plots()
    run_normals_comparison()