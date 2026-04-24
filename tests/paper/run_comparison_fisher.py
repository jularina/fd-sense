import matplotlib.pyplot as plt
from hydra.utils import instantiate, get_original_cwd
import hydra
from omegaconf import DictConfig
import os
import glob
import numpy as np
import pymc as pm
from dataclasses import dataclass

from src.plots.paper.comparison_paper_funcs import *
from src.utils.files_operations import load_plot_config
from src.optimization.corner_points_fisher import *
from src.bayesian_model.base import BayesianModel
from src.discrepancies.posterior_fisher import PosteriorFDBase
from src.discrepancies.prior_fisher import PriorFDNonParametric
from src.discrepancies.posterior_fisher import PosteriorFDNonParametric
from src.optimization.nonparametric_fisher import OptimisationNonparametricBase
from src.plots.paper.toy_paper_fisher_funcs import *
from src.basis_functions.basis_functions import *


@dataclass
class BlockDecomposition:
    name: str
    main_quad: float          # eta_j^T A_jj eta_j
    linear: float             # b_j^T eta_j
    interaction: float        # sum_{k!=j} eta_j^T A_jk eta_k   (counted ONCE per block; totals will be 2x this)
    total_with_half_interactions: float  # main + linear + interaction/?? (see note below)


def decompose_prior_qf_by_blocks(
    eta: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    c: float,
    names: List[str],
    block_size: int = 2,
) -> Tuple[Dict[str, BlockDecomposition], Dict[str, float]]:
    """
    Decompose Q(eta)=eta^T A eta + b^T eta + c into block contributions.

    Returns:
      per_block: dict of BlockDecomposition
      totals: dict with sanity-check totals
    """
    eta = np.asarray(eta).reshape(-1)
    b = np.asarray(b).reshape(-1)
    A = np.asarray(A)

    d = eta.shape[0]
    assert A.shape == (d, d), f"A shape {A.shape} does not match eta length {d}"
    assert b.shape == (d,), f"b length {b.shape} does not match eta length {d}"
    assert d == block_size * len(names), f"eta length {d} != {block_size} * len(names) {len(names)}"

    # Full value
    Q_full = float(eta @ A @ eta + b @ eta + c)
    per_block: Dict[str, BlockDecomposition] = {}

    # Precompute block indices
    block_inds = []
    for j, nm in enumerate(names):
        idx = np.arange(j * block_size, (j + 1) * block_size)
        block_inds.append((nm, idx))

    # Compute within-block and interaction pieces
    for j, (nm, idx_j) in enumerate(block_inds):
        eta_j = eta[idx_j]
        b_j = b[idx_j]
        A_jj = A[np.ix_(idx_j, idx_j)]

        main_quad = float(eta_j @ A_jj @ eta_j)
        linear = float(b_j @ eta_j)

        # interaction_j := sum_{k != j} eta_j^T A_jk eta_k  (no factor 2 here)
        interaction = 0.0
        for k, (_, idx_k) in enumerate(block_inds):
            if k == j:
                continue
            A_jk = A[np.ix_(idx_j, idx_k)]
            eta_k = eta[idx_k]
            interaction += float(eta_j @ A_jk @ eta_k)

        # A clean “per-block total” that sums back to Q (up to constant) is:
        # main + linear + 0.5 * (2 * sum_{j<k} eta_j^T A_jk eta_k) attributed evenly.
        # Since interaction here counts both directions across j, we use 0.5 * interaction.
        total_with_half_interactions = main_quad + linear + 0.5 * interaction

        per_block[nm] = BlockDecomposition(
            name=nm,
            main_quad=main_quad,
            linear=linear,
            interaction=interaction,
            total_with_half_interactions=total_with_half_interactions,
        )

    # Sanity checks
    main_sum = sum(v.main_quad for v in per_block.values())
    linear_sum = sum(v.linear for v in per_block.values())

    # Each cross term eta_j^T A_jk eta_k is counted twice in sum_j interaction_j (once as j→k and once as k→j),
    # so the true total cross contribution to eta^T A eta is:
    # cross_total = 0.5 * sum_j interaction_j
    cross_total = 0.5 * sum(v.interaction for v in per_block.values())

    Q_reconstructed_no_c = main_sum + cross_total + linear_sum
    Q_reconstructed = Q_reconstructed_no_c + float(c)

    totals = {
        "Q_full": Q_full,
        "main_sum": float(main_sum),
        "linear_sum": float(linear_sum),
        "cross_total": float(cross_total),
        "c": float(c),
        "Q_reconstructed": float(Q_reconstructed),
        "abs_err": float(abs(Q_reconstructed - Q_full)),
    }

    return per_block, totals


def rank_blocks(per_block: Dict[str, BlockDecomposition], key: str = "total_with_half_interactions"):
    """Return a list of (name, value) sorted descending by chosen key."""
    pairs = [(nm, getattr(obj, key)) for nm, obj in per_block.items()]
    return sorted(pairs, key=lambda x: x[1], reverse=True)


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


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="comparison_section_1_4_berger")
def run_laplace_cauchy_priors_uniform_reference_1_4_berger(cfg: DictConfig, save_samples: bool = False) -> None:
    """
    Same as the 'normal reference' runner, but creates reference posterior samples
    assuming a Uniform(a,b) prior. With one observation x and N(x|theta,1) likelihood,
    the posterior is N(x,1) truncated to [a,b].
    """
    base_dir = os.path.join(get_original_cwd(), "data", "comparison", "section_1_4_berger")

    if save_samples:
        rng_seed = 27
        n_samples = 5000
        rng = np.random.default_rng(rng_seed)
        a, b = -5.0, 5.0

        x_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        def _tag(x):
            return str(x).replace(".", "_")

        for x in x_values:
            samples = sample_trunc_normal_loc1(x=float(x), a=a, b=b, n=n_samples, rng=rng)
            np.save(os.path.join(base_dir, f"posterior_samples_uniform_reference_x_{_tag(x)}.npy"), samples)

    obs_files = sorted(glob.glob(os.path.join(base_dir,  "obs_x_*.npy")))
    post_files = sorted(glob.glob(os.path.join(base_dir, "posterior_samples_uniform_reference_x_*.npy")))
    assert len(obs_files) == len(post_files), "Mismatch between obs and posterior sample files."

    for obs_path, post_path in zip(obs_files, post_files):
        cfg.data.observations_path = obs_path
        cfg.data.posterior_samples_path = post_path
        model = instantiate(cfg.model, data_config=cfg.data)
        fisher_estimator = PosteriorFDBase(samples=model.posterior_samples_init, model=model, candidate_type="prior")
        fd_value = float(fisher_estimator.estimate_fisher())
        x_val = float(np.load(obs_path).item())
        print(f"x = {x_val:.1f} | Initial FD (Uniform ref): {fd_value:.4f}")


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/toy/", config_name="comparison_section_1_4_berger")
def run_laplace_cauchy_priors_normal_reference_1_4_berger(cfg, save_samples=False) -> None:
    """
    Main function to compute KSD and perform prior parameter grid search using Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    base_dir = os.path.join(get_original_cwd(), "data", "comparison", "section_1_4_berger")

    if save_samples:
        rng_seed = 27
        n_samples = 5000
        np.random.seed(rng_seed)

        mu0 = 0.0
        sigma0 = 1.414
        sigma_noise2 = 1.0

        x_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        def _tag(x):
            return str(x).replace(".", "_")

        for x in x_values:
            obs_arr = np.array([x], dtype=float)
            np.save(os.path.join(base_dir, f"obs_x_{_tag(x)}.npy"), obs_arr)
            tau0 = 1.0 / (sigma0 ** 2)
            tau_noise = 1.0 / sigma_noise2
            tau_post = tau0 + tau_noise

            mu_post = (tau0 * mu0 + tau_noise * x) / tau_post
            sigma_post2 = 1.0 / tau_post
            sigma_post = np.sqrt(sigma_post2)
            samples = np.random.normal(loc=mu_post, scale=sigma_post, size=(n_samples, 1))
            np.save(os.path.join(base_dir, f"posterior_samples_normal_reference_x_{_tag(x)}.npy"), samples)

    # Instantiate the Bayesian model with runtime overrides
    obs_files = sorted(glob.glob(os.path.join(base_dir, "obs_x_*.npy")))
    post_files = sorted(glob.glob(os.path.join(base_dir, "posterior_samples_normal_reference_x_*.npy")))
    assert len(obs_files) == len(post_files), "Mismatch between obs and posterior sample files."

    for obs_path, post_path in zip(obs_files, post_files):
        cfg.data.observations_path = obs_path
        cfg.data.posterior_samples_path = post_path
        model = instantiate(cfg.model, data_config=cfg.data)
        fisher_estimator = PosteriorFDBase(samples=model.posterior_samples_init, model=model, candidate_type="prior")
        fd_value = float(fisher_estimator.estimate_fisher())
        obs_val = float(np.load(obs_path).item())
        print(f"x = {obs_val:.1f} | Initial FD: {fd_value:.4f}")


def _pymc_posterior_samples_ref(x_obs: float,
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
            x = float(prior_params.get("x"))
            gamma = float(prior_params.get("gamma"))
            theta = pm.Cauchy("theta", alpha=x, beta=gamma)
        elif prior_kind == "uniform":
            a = float(prior_params.get("a"))
            b = float(prior_params.get("b"))
            theta = pm.Uniform("theta", lower=a, upper=b)
        else:
            raise ValueError("prior_kind must be one of {'laplace', 'cauchy', 'uniform'}")

        pm.Normal("x_obs", mu=theta, sigma=1.0, observed=float(x_obs))

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
        return f"laplace_mu{prior_params.get('mu', 0.0)}_b{prior_params.get('b', 1.0)}"
    if pk == "cauchy":
        return f"cauchy_mu{prior_params.get('mu', 0.0)}_g{prior_params.get('gamma', 1.0)}"
    if pk == "uniform":
        return f"uniform_{prior_params.get('a', -5.0)}_{prior_params.get('b', 5.0)}"
    return "unknown_ref"


@hydra.main(version_base="1.1",
            config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="comparison_section_1_4_berger")
def run_nonconjugate_reference_prior_mcmc_1_4_berger(cfg: DictConfig, save_samples: bool = False) -> None:
    base_dir = os.path.join(get_original_cwd(), "data", "comparison", "section_1_4_berger")
    prior_kind = 'cauchy'
    prior_params = {'x': 0, 'gamma': 4}
    x_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]

    draws = int(getattr(cfg.data, "ref_draws", 5000))
    tune = int(getattr(cfg.data, "ref_tune", 2000))
    target_accept = float(getattr(cfg.data, "ref_target_accept", 0.95))
    seed = int(getattr(cfg.data, "ref_seed", 27))

    if save_samples:
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        def _tagx(x): return str(x).replace(".", "_")

        for x in x_values:
            samples = _pymc_posterior_samples_ref(
                x_obs=x,
                prior_kind=prior_kind,
                prior_params=prior_params,
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                random_seed=seed,
            )
            np.save(os.path.join(base_dir, f"posterior_samples_cauchy_reference_x_{_tagx(x)}.npy"),
                    samples)

    obs_files = sorted(glob.glob(os.path.join(base_dir,  "obs_x_*.npy")))
    post_files = sorted(glob.glob(os.path.join(base_dir, "posterior_samples_cauchy_reference_x_*.npy")))
    assert len(obs_files) == len(post_files), "Mismatch between obs and posterior sample files."

    for obs_path, post_path in zip(obs_files, post_files):
        cfg.data.observations_path = obs_path
        cfg.data.posterior_samples_path = post_path
        x_val = float(np.load(obs_path).item())
        model = instantiate(cfg.model, data_config=cfg.data)
        fisher_estimator = PosteriorFDBase(samples=model.posterior_samples_init, model=model, candidate_type="prior")
        fd_value = float(fisher_estimator.estimate_fisher())
        print(f"[{prior_kind} ref] x = {x_val:.1f} | Initial FD: {fd_value:.4f}")


@hydra.main(version_base="1.1",
            config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="comparison_section_1_4_berger")
def plot_comparison_plots_1_4_berger(cfg: DictConfig):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    plot_cfg = load_plot_config(plot_config_path)
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)

    # 1) Reference-based shape plots
    plot_reference_shapes_x(3.0, "normal", plot_cfg, output_dir)
    plot_reference_shapes_x(3.0, "uniform", plot_cfg, output_dir)
    plot_reference_shapes_x(3.0, "cauchy", plot_cfg, output_dir)

    # 2) Informal approach plot
    means = {
        r"$E[\theta|x]$ (Normal prior): 2.0": 2.0,
        r"$E[\theta|x]$ (Laplace prior): 2.29": 2.29,
        r"$E[\theta|x]$ (Cauchy prior): 2.27": 2.27,
    }
    plot_informal_means_x(3.0, means, plot_cfg, output_dir)


@hydra.main(version_base="1.1",
            config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="comparison_section_3_3_berger")
def run_normals_comparison_section_3_3_berger(cfg: DictConfig, save_samples: bool = True):
    base_dir = os.path.join(get_original_cwd(), "data", "comparison", "section_3_3_2_berger")

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
            np.save(os.path.join(base_dir, f"obs_x_{_tag2(x1, x2)}.npy"), x.reshape((-1, 2)))
            mu_post = 0.5 * x
            samples = np.random.multivariate_normal(mean=mu_post, cov=Sigma_post, size=n_samples)
            np.save(os.path.join(base_dir, f"posterior_samples_x_{_tag2(x1, x2)}.npy"), samples)

    obs_files = sorted(glob.glob(os.path.join(base_dir, "obs_x_*.npy")))
    post_files = sorted(glob.glob(os.path.join(base_dir, "posterior_samples_x_*.npy")))
    assert len(obs_files) == len(post_files), "Mismatch between obs and posterior sample files."

    for obs_path, post_path in zip(obs_files, post_files):
        cfg.data.observations_path = obs_path
        cfg.data.posterior_samples_path = post_path
        model = instantiate(cfg.model, data_config=cfg.data)
        fisher_estimator = PosteriorFDBase(samples=model.posterior_samples_init, model=model, candidate_type="prior")
        fd_value = float(fisher_estimator.estimate_fisher())
        obs_val = np.load(obs_path)
        print(f"(x_0, x_1) = {obs_val} | Initial FD: {fd_value:.4f}")


def sample_bioassay(x, y, n):
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / std * 0.5

    ngrid = 100
    A = np.linspace(-4, 8, ngrid)
    B = np.linspace(-10, 40, ngrid)
    ilogit_abx = 1 / (np.exp(-(A[:, None] + B[:, None, None] * x)) + 1)
    p = np.prod(ilogit_abx ** y * (1 - ilogit_abx) ** (n - y), axis=2)
    nsamp = 2000
    samp_indices = np.unravel_index(
        np.random.choice(p.size, size=nsamp, p=p.ravel() / np.sum(p)),
        p.shape
    )
    samp_A = A[samp_indices[1]]
    samp_B = B[samp_indices[0]]
    samp_A += (np.random.rand(nsamp) - 0.5) * (A[1] - A[0])
    samp_B += (np.random.rand(nsamp) - 0.5) * (B[1] - B[0])
    samp_ld50 = -samp_A / samp_B

    plt.hist(samp_ld50)
    plt.show()

    plt.scatter(samp_A, samp_B)
    plt.show()

    return np.column_stack((samp_A, samp_B)), samp_ld50, x


@hydra.main(version_base="1.1",
            config_path="../../configs/paper/ksd_calculation/real/",
            config_name="bioassay_model")
def run_bioassay(cfg: DictConfig, sample=False):
    data_dir = os.path.join(get_original_cwd(), "data/bioassay/")
    x = np.load(data_dir + "x.npy")
    y = np.load(data_dir + "y.npy")
    n = np.load(data_dir + "n.npy")

    if sample:
        samples, samples_ld50, x = sample_bioassay(x=x, n=n, y=y)
        np.save(data_dir + "samples.npy", samples)
        np.save(data_dir + "samples_ld50.npy", samples_ld50)
        observations = np.column_stack((x, y, n))
        np.save(data_dir + "observations.npy", observations)

    print("=== FD for Bioassay model ===")
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    fd_value = float(fisher_estimator.estimate_fisher_prior_only())
    print(f"[FD] Posterior FD (baseline prior): {fd_value:.5f}")

    # Parametric prior optimisation
    optimizer = OptimizationCornerPointsCompositePrior(fisher_estimator,
                                                       cfg.fd.optimize.prior.Composite,
                                                       cfg.fd.optimize.loss.LogisticBinomialLogLikelihood,
                                                       )
    qf_corners, eta_star = optimizer.evaluate_all_prior_corners()

    # Interpretation
    A, b, c = fisher_estimator.compute_fisher_quadratic_form_prior_only()
    names = ["alpha", "beta"]
    per_block, totals = decompose_prior_qf_by_blocks(
        eta=eta_star, A=A, b=b, c=c, names=names, block_size=2
    )
    print(totals)
    ranking = rank_blocks(per_block, key="total_with_half_interactions")
    print("Ranked (main + linear + 0.5*interactions):")
    for nm, val in ranking:
        print(nm, val)


@hydra.main(version_base="1.1",
            config_path="../../configs/paper/ksd_calculation/toy/",
            config_name="comparison_section_1_4_berger")
def plot_comparison_plots_wim(cfg: DictConfig):
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    plot_cfg = load_plot_config(plot_config_path)
    output_dir = os.path.join(get_original_cwd(), "outputs/paper/plots/comparison/wim/")

    plot_prior_range_comparison_split(
        wim_cauchy_scales_beta=[2.5, 5, 10],
        wim_cauchy_scales_alpha=[10],
        eta_1_alpha_range=[-0.12, 0.12],
        eta_2_alpha_range=[-0.02, -0.00125],
        eta_1_beta_range=[-5.12, 5.12],
        eta_2_beta_range=[-1.28, -0.005],
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        filenames=(
            "comparison_prior_range_wim_cauchy.pdf",
            "comparison_prior_range_wim_fd_normal.pdf",
        ),
        normal_fill_alpha=0.12
    )


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _cauchy_pdf(x, loc=0.0, scale=1.0):
    # density of Cauchy(loc, scale)
    z = (x - loc) / scale
    return 1.0 / (np.pi * scale * (1.0 + z**2))


def sample_ecmo(
    y1=6, n1=10,
    y2=9, n2=9,
    gamma_loc=0.0, gamma_scale=0.419,
    delta_loc=0.0, delta_scale=1.099,
    ngrid_gamma=600,
    ngrid_delta=600,
    gamma_range=(-8.0, 8.0),
    delta_range=(-12.0, 12.0),
    nsamp=2000,
    seed=0,
):
    """
    Sample approximately from the reference posterior for ECMO model using a grid approximation.

    Reference prior:
        gamma ~ Cauchy(0, 0.419)
        delta ~ Cauchy(0, 1.099)

    Parameters are:
        eta1 = gamma - delta/2
        eta2 = gamma + delta/2
        p1 = sigmoid(eta1), p2 = sigmoid(eta2)

    Likelihood:
        y1 ~ Bin(n1, p1), y2 ~ Bin(n2, p2)
    """
    rng = np.random.default_rng(seed)

    # Grid in (gamma, delta)
    gammas = np.linspace(gamma_range[0], gamma_range[1], ngrid_gamma)
    deltas = np.linspace(delta_range[0], delta_range[1], ngrid_delta)

    G, D = np.meshgrid(gammas, deltas, indexing="xy")  # shapes (ngrid_delta, ngrid_gamma)

    # Map (gamma, delta) -> (p1, p2)
    eta1 = G - 0.5 * D
    eta2 = G + 0.5 * D
    p1 = _sigmoid(eta1)
    p2 = _sigmoid(eta2)

    # Log-likelihood up to an additive constant (binomial coefficients drop)
    # log L = y1 log p1 + (n1-y1) log(1-p1) + y2 log p2 + (n2-y2) log(1-p2)
    eps = 1e-12
    loglik = (
        y1 * np.log(np.clip(p1, eps, 1 - eps)) + (n1 - y1) * np.log(np.clip(1 - p1, eps, 1 - eps)) +
        y2 * np.log(np.clip(p2, eps, 1 - eps)) + (n2 - y2) * np.log(np.clip(1 - p2, eps, 1 - eps))
    )

    # Log-prior (independent Cauchy)
    prior = _cauchy_pdf(G, loc=gamma_loc, scale=gamma_scale) * _cauchy_pdf(D, loc=delta_loc, scale=delta_scale)
    logprior = np.log(np.clip(prior, eps, None))

    # Unnormalised log-posterior on the grid
    logpost = loglik + logprior

    # Stabilise and convert to probabilities
    logpost = logpost - np.max(logpost)
    post_unnorm = np.exp(logpost)
    post = post_unnorm / np.sum(post_unnorm)

    # Sample grid cells according to posterior mass
    flat_idx = rng.choice(post.size, size=nsamp, replace=True, p=post.ravel())
    idx_delta, idx_gamma = np.unravel_index(flat_idx, post.shape)

    samp_gamma = gammas[idx_gamma].astype(float)
    samp_delta = deltas[idx_delta].astype(float)

    # Jitter within cell to avoid grid artifacts
    dg = (gammas[1] - gammas[0]) if ngrid_gamma > 1 else 1.0
    dd = (deltas[1] - deltas[0]) if ngrid_delta > 1 else 1.0
    samp_gamma += (rng.random(nsamp) - 0.5) * dg
    samp_delta += (rng.random(nsamp) - 0.5) * dd

    samples = np.column_stack((samp_gamma, samp_delta))

    return samples


def sample_ecmo_prior(
    nsamp=2000,
    gamma_loc=0.0, gamma_scale=0.419,
    delta_loc=0.0, delta_scale=1.099,
    seed=0,
):
    """Independent Cauchy prior samples for (gamma, delta)."""
    rng = np.random.default_rng(seed)
    samp_gamma = rng.standard_cauchy(size=nsamp) * gamma_scale + gamma_loc
    samp_delta = rng.standard_cauchy(size=nsamp) * delta_scale + delta_loc
    return np.column_stack((samp_gamma, samp_delta))


# Interpretability
def log_cauchy_pdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    z = (x - loc) / scale
    return -np.log(np.pi * scale) - np.log1p(z**2)


def log_gaussian_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    # x: (m,d)
    x = np.asarray(x, dtype=float)
    d = x.shape[1]
    mean = np.asarray(mean, dtype=float).reshape(1, d)
    cov = np.asarray(cov, dtype=float)
    L = np.linalg.cholesky(cov)
    diff = x - mean
    sol = np.linalg.solve(L, diff.T)          # (d,m)
    quad = np.sum(sol**2, axis=0)             # (m,)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2*np.pi) + logdet + quad)


def log_pi_ref(samples: np.ndarray,
               gamma_loc=0.0, gamma_scale=0.419,
               delta_loc=0.0, delta_scale=1.099) -> np.ndarray:
    gamma = samples[:, 0]
    delta = samples[:, 1]
    return log_cauchy_pdf(gamma, gamma_loc, gamma_scale) + log_cauchy_pdf(delta, delta_loc, delta_scale)


def log_pi_K_unnormalized(samples: np.ndarray,
                          eta: np.ndarray,
                          basis_func,
                          g_mean: np.ndarray,
                          g_cov: np.ndarray) -> np.ndarray:
    """
    Unnormalised log candidate prior: eta^T T(theta) + log g(theta).
    (Normaliser cancels in self-normalised IS.)
    """
    eta = np.asarray(eta, dtype=float).reshape(-1)  # (K,)
    # basis evaluate
    Phi = basis_func.evaluate(samples)  # (m,d,K)
    if Phi.ndim != 3:
        raise ValueError(f"Expected basis_func.evaluate to return (m,d,K), got {Phi.shape}")
    # Use scalar basis per k. Prefer metric='full' so Phi[:,0,:] is the scalar basis.
    T = Phi[:, 0, :]  # (m,K)
    log_g = log_gaussian_pdf(samples, mean=g_mean, cov=g_cov)  # (m,)
    return T @ eta + log_g


def prob_delta_positive_under_eta(
    posterior_samples: np.ndarray,
    eta_star: np.ndarray,
    basis_func,
    g_mean: np.ndarray,
    g_cov: np.ndarray,
    gamma_loc=0.0, gamma_scale=0.419,
    delta_loc=0.0, delta_scale=1.099,
) -> float:
    """
    Self-normalised IS estimate of P(delta>0 | y) under candidate prior defined by eta_star.
    posterior_samples are drawn from reference posterior.
    """
    logw = (
        log_pi_K_unnormalized(posterior_samples, eta_star, basis_func, g_mean, g_cov)
        - log_pi_ref(posterior_samples, gamma_loc, gamma_scale, delta_loc, delta_scale)
    )
    # stabilise
    logw = logw - np.max(logw)
    w = np.exp(logw)
    ind = (posterior_samples[:, 1] > 0).astype(float)
    return float(np.sum(w * ind) / np.sum(w))


def compute_interpretability(cfg, model, result_sdp, fd_estimates_list, radii):
    posterior_samples = model.posterior_samples_init
    prior_samples = model.prior_samples_init
    basis_func = MaternBasisFunctionMultidim(
        posterior_samples=posterior_samples,
        prior_samples=prior_samples,
        num_basis_functions=cfg.ksd.optimize.prior.nonparametric.basis_funcs_kwargs["num_basis_functions"],
        metric="full",
        method="quantile_grid",
        estimation_samples_source="posterior",
        estimation_centers_source="posterior",
    )
    g_mean = np.zeros(2)
    g_cov = np.cov(posterior_samples, rowvar=False) + 1e-3 * np.eye(2)
    prob_list = []

    for i, (eta_star, radius) in enumerate(zip(result_sdp, radii)):
        p_delta_pos = prob_delta_positive_under_eta(
            posterior_samples=posterior_samples,
            eta_star=eta_star,
            basis_func=basis_func,
            g_mean=g_mean,
            g_cov=g_cov,
        )
        prob_list.append(p_delta_pos)
        print(f"radius={radius:>5}  FD*={fd_estimates_list[i]:.4f}  P(delta>0)={p_delta_pos:.4f}")


@hydra.main(version_base="1.1",
            config_path="../../configs/paper/ksd_calculation/real/",
            config_name="ecmo_model")
def run_ecmo(cfg: DictConfig, sample=False):
    data_dir = os.path.join(get_original_cwd(), "data/ecmo/")
    y1, n1 = 6, 10
    y2, n2 = 9, 9

    if sample:
        samples = sample_ecmo(
            y1=y1, n1=n1,
            y2=y2, n2=n2,
            gamma_loc=0.0, gamma_scale=0.419,
            delta_loc=0.0, delta_scale=1.099,
            ngrid_gamma=600,
            ngrid_delta=600,
            gamma_range=(-8.0, 8.0),
            delta_range=(-12.0, 12.0),
            nsamp=5000,
            seed=0,
        )
        prior_samples = sample_ecmo_prior(
            nsamp=2000,
            gamma_loc=0.0, gamma_scale=0.419,
            delta_loc=0.0, delta_scale=1.099,
            seed=1,
        )
        np.save(os.path.join(data_dir, "prior_samples.npy"), prior_samples)
        np.save(os.path.join(data_dir, "posterior_samples.npy"), samples)
        observations = np.array([[y1, n1], [y2, n2]], dtype=float)
        np.save(os.path.join(data_dir, "observations.npy"), observations)

    print("=== FD for ECMO model ===")
    model: BayesianModel = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    fd_value = float(fisher_estimator.estimate_fisher_prior_only())
    print(f"[FD] Posterior FD (baseline prior): {fd_value:.5f}")

    # Nonparametric optimisation
    radii = [1.5, 3.0, 5.0, 10.0, 20.0, 40.0]
    estimator_prior = PriorFDNonParametric(model=model)
    estimator_posterior = PosteriorFDNonParametric(model=model)
    psi_sdp_list, fd_estimates_list, radius_labels = [], [], []

    for radius in radii:
        optimizer = OptimisationNonparametricBase(
            estimator_posterior,
            estimator_prior,
            cfg.ksd.optimize.prior.nonparametric,
            radius=radius,
            add_nuggets=False,
        )
        result_sdp = optimizer.optimize_through_sdp_relaxation()
        psi_sdp_list.append(result_sdp["eta_star"])
        fd_estimates_list.append(result_sdp["primal_value"])
        radius_labels.append(radius)

    compute_interpretability(cfg, model, psi_sdp_list, fd_estimates_list, radii)

    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)
    plot_sdp_2d_densities(
        basis_function=optimizer.basis_function,
        psi_sdp_list=psi_sdp_list,
        radius_labels=radius_labels,
        ksd_estimates=fd_estimates_list,
        prior_distribution=model.prior_init,
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        domain=((-5, 5), (-7, 15)),
        resolution=300
    )


if __name__ == "__main__":
    # run_laplace_cauchy_priors_normal_reference_1_4_berger()
    # run_laplace_cauchy_priors_uniform_reference_1_4_berger()
    # run_nonconjugate_reference_prior_mcmc_1_4_berger()
    # plot_comparison_plots_1_4_berger()
    # run_normals_comparison_section_3_3_berger()
    run_bioassay()
    # plot_comparison_plots_wim()
    # run_ecmo()
