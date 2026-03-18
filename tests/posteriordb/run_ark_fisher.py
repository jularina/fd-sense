import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import os
import warnings
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import time
import json

from src.discrepancies.posterior_fisher import PosteriorFDBase
from src.utils.files_operations import load_plot_config
from src.plots.paper.posterior_db_paper_funcs import *
from src.optimization.corner_points_fisher import (
    OptimizationCornerPointsCompositePrior
)
from src.plots.paper.sbi_paper_funcs import *

warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


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


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ark_posteriordb")
def main(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    print("=== FD for PosteriorDB model ===")
    model = instantiate(cfg.model, data_config=cfg.data)
    fisher_estimator = PosteriorFDBase(model=model)
    print(f"Initial Fisher for prior: {fisher_estimator.estimate_fisher_prior_only():.4f}")
    print(f"Initial Fisher for lr: {fisher_estimator.estimate_fisher_lr_only():.4f}")

    print("Starting optimisation of all parameters at once.")
    start = time.perf_counter()
    optimizer = OptimizationCornerPointsCompositePrior(fisher_estimator,
                                                       cfg.fd.optimize.prior.Composite,
                                                       cfg.fd.optimize.loss.GaussianARLogLikelihood,
                                                       )
    qf_corners, eta_star = optimizer.evaluate_all_prior_corners()
    # elapsed = time.perf_counter() - start
    # print(f"Time for optimisation of all parameters at once: {elapsed:.3f} sec.")
    #
    # print("Starting black-box optimisation.")
    # start = time.perf_counter()
    # bb = optimizer.black_box_optimize_prior_box_global(
    #     seed=0,
    #     maxiter=150,
    #     popsize=20,
    #     workers=1,
    #     updating="deferred",
    # )
    # elapsed = time.perf_counter() - start
    # print("Black-box sup:", bb.val_sup)
    # print("Black-box inf:", bb.val_inf)
    # print("Black-box S_hat:", bb.S_hat)
    # print(f"Time for black-box optimisation of all parameters at once: {elapsed:.3f} sec.")
    #
    # print("Starting per component optimisation.")
    # names = ["alpha", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma"]
    # start = time.perf_counter()
    # sup_res, eta_sup_blocks = optimizer.evaluate_all_prior_corners_per_component(component_names=names)
    # eta_inf_blocks, values_inf = optimizer.minimize_prior_per_component_qp(names)
    # print("Per-component argmax corners:")
    # for k in names:
    #     print(k, eta_sup_blocks[k], sup_res[k][0][1])
    #
    # print("Per-component infimum:")
    # for n in names:
    #     print(n, eta_inf_blocks[n], values_inf[n])
    # elapsed = time.perf_counter() - start
    # print(f"Time for per-component optimisation: {elapsed/len(names):.3f} sec.")
    #
    # # Interpretation
    # A, b, c = fisher_estimator.compute_fisher_quadratic_form_prior_only()
    # names = ["alpha", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma"]
    # per_block, totals = decompose_prior_qf_by_blocks(
    #     eta=eta_star, A=A, b=b, c=c, names=names, block_size=2
    # )
    # print(totals)
    # ranking = rank_blocks(per_block, key="total_with_half_interactions")
    # print("Ranked (main + linear + 0.5*interactions):")
    # for nm, val in ranking:
    #     print(nm, val)
    #
    # eta_sigma_inf = eta_inf_blocks["sigma"]
    # sigma_inf = {
    #     "family": "Gamma",
    #     "params": {
    #         "alpha": float(eta_sigma_inf[0] + 1.0),
    #         "theta": float(-1.0 / eta_sigma_inf[1]),
    #     },
    # }
    #
    # alpha_ref = {"family": "Gaussian", "params": {"mu": 0.0, "sigma": 10.0}}
    # betas_ref = {"family": "Gaussian", "params": {"mu": 0.0, "sigma": 10.0}}
    # sigma_ref = {"family": "HalfCauchy", "params": {"gamma": 2.5}}
    #
    # alpha_ms = {"family": "Gaussian", "params": {"mu": 3.0, "sigma": 0.4}}
    # betas_ms = {
    #     "beta1": {"family": "Gaussian", "params": {"mu": -3.0, "sigma": 0.4}},
    #     "beta2": {"family": "Gaussian", "params": {"mu": -3.0, "sigma": 0.4}},
    #     "beta3": {"family": "Gaussian", "params": {"mu": -3.0, "sigma": 0.4}},
    #     "beta4": {"family": "Gaussian", "params": {"mu": 3.0, "sigma": 0.4}},
    #     "beta5": {"family": "Gaussian", "params": {"mu": 3.0, "sigma": 0.4}},
    # }
    # sigma_ms = {
    #     "family": "Gamma",
    #     "params": {"alpha": 11.11, "theta": 1.0 / 22.22},
    # }
    # alpha_box_ranges = {
    #     "mu": (-3.0, 3.0),
    #     "sigma": (0.4, 1.0),
    # }
    #
    # beta_box_ranges_neg = {
    #     "mu": (-3.0, 0.0),
    #     "sigma": (0.4, 1.0),
    # }
    #
    # beta_box_ranges_pos = {
    #     "mu": (0.0, 3.0),
    #     "sigma": (0.4, 1.0),
    # }
    # betas_box_ranges = {
    #     "beta1": beta_box_ranges_neg,
    #     "beta2": beta_box_ranges_neg,
    #     "beta3": beta_box_ranges_neg,
    #     "beta4": beta_box_ranges_pos,
    #     "beta5": beta_box_ranges_pos,
    # }
    # sigma_box_ranges = {
    #     "alpha": (4.0, 11.11),
    #     "theta": (1.0 / 22.22, 0.5),
    # }
    #
    # plot_three_panel_priors_all_betas_one_plot_explicit(
    #     alpha_ref=alpha_ref,
    #     betas_ref=betas_ref,
    #     sigma_ref=sigma_ref,
    #     alpha_ms=alpha_ms,
    #     betas_ms=betas_ms,
    #     sigma_ms=sigma_ms,
    #     sigma_inf=sigma_inf,
    #     alpha_box_ranges=alpha_box_ranges,
    #     betas_box_ranges=betas_box_ranges,
    #     sigma_box_ranges=sigma_box_ranges,
    #     plot_cfg=plot_cfg,
    #     output_dir=output_dir,
    #     prefix=prefix,
    #     sample_n_alpha=50,
    #     sample_n_sigma=30,
    #     sample_n_beta_total=150,
    #     seed=27,
    #     filename="ark_param_three_panel_priors.pdf",
    # )


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ark_posteriordb")
def compare_complexities(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    # Replace with the measured times (seconds)
    qf_full_time_sec = 0.093
    qf_decomp_time_sec = 0.008
    black_box_time_sec = 154.0

    plot_complexity_bar(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        filename="ark_computational_cost.pdf",
        use_log10=True,
        qf_full_time_sec=qf_full_time_sec,
        qf_decomp_time_sec=qf_decomp_time_sec,
        black_box_time_sec=black_box_time_sec,
    )


def _load_corner_draws_json(path: str) -> list[dict]:
    """
    Expected JSON structure:
      list of length n_chains (e.g. 10)
      each element is a dict with keys like:
        'alpha', 'sigma', 'beta[1]'...'beta[5]'  (or possibly 'beta' as vector)
      each value is an array-like of draws (e.g. 10000)
    """
    with open(path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a list of chains, got {type(obj)}")
    if len(obj) == 0:
        raise ValueError("No chains in corner draws JSON.")
    return obj


def _norm_warmup(warmup, n_chains: int) -> list[int]:
    if isinstance(warmup, (list, tuple)):
        if len(warmup) != n_chains:
            raise ValueError(f"warmup length {len(warmup)} != n_chains {n_chains}")
        return [max(0, int(w)) for w in warmup]
    return [max(0, int(warmup))] * n_chains


def _stack_corner_chains(
    chains: list[dict],
    K: int,
    warmup=0,
    max_draws: int | None = None,
) -> np.ndarray:
    """
    Returns samples as (N, 2+K) = (N, [alpha, beta_1..beta_K, sigma]) in that order.
    Supports chain dict keys either:
      - alpha, sigma, beta[1]..beta[K]
      - alpha, sigma, beta (vector of shape (n_draws, K) or (n_draws, ..))
    """
    n_chains = len(chains)
    warmups = _norm_warmup(warmup, n_chains)

    per_chain_mats = []
    for ci, d in enumerate(chains):
        # ---- get alpha
        if "alpha" not in d:
            raise ValueError(f"Chain {ci} missing 'alpha' key. Keys: {list(d.keys())}")
        alpha = np.asarray(d["alpha"], dtype=float).reshape(-1)

        # ---- get sigma
        if "sigma" not in d:
            raise ValueError(f"Chain {ci} missing 'sigma' key. Keys: {list(d.keys())}")
        sigma = np.asarray(d["sigma"], dtype=float).reshape(-1)

        # ---- get betas
        beta_cols = []
        if "beta" in d:
            beta = np.asarray(d["beta"], dtype=float)
            if beta.ndim == 1:
                # if stored as flat vector per draw (unlikely), infer K
                if beta.shape[0] != alpha.shape[0] * K:
                    raise ValueError(f"Chain {ci}: cannot reshape beta of shape {beta.shape} to (n_draws,K).")
                beta = beta.reshape(alpha.shape[0], K)
            elif beta.ndim == 2:
                if beta.shape[1] != K:
                    raise ValueError(f"Chain {ci}: beta has shape {beta.shape}, expected K={K}.")
            else:
                raise ValueError(f"Chain {ci}: unsupported beta ndim={beta.ndim}.")
            beta_cols = [beta[:, j] for j in range(K)]
        else:
            # beta[1],...,beta[K]
            for s in range(1, K + 1):
                key = f"beta[{s}]"
                if key not in d:
                    raise ValueError(f"Chain {ci} missing '{key}' key. Keys: {list(d.keys())}")
                beta_cols.append(np.asarray(d[key], dtype=float).reshape(-1))

        # ---- sanity: same length
        n = alpha.shape[0]
        if sigma.shape[0] != n or any(b.shape[0] != n for b in beta_cols):
            raise ValueError(f"Chain {ci}: parameter draw lengths mismatch.")

        # ---- trim warmup and optional max_draws
        w = min(warmups[ci], n)
        sl = slice(w, None)
        alpha_t = alpha[sl]
        sigma_t = sigma[sl]
        beta_t = [b[sl] for b in beta_cols]

        if max_draws is not None:
            alpha_t = alpha_t[:max_draws]
            sigma_t = sigma_t[:max_draws]
            beta_t = [b[:max_draws] for b in beta_t]

        mat = np.column_stack([alpha_t] + beta_t + [sigma_t])
        per_chain_mats.append(mat)

    return np.concatenate(per_chain_mats, axis=0)


# -----------------------------
# Posterior predictive
# -----------------------------

def _ar_posterior_predictive(
    y_full: np.ndarray,
    samples: np.ndarray,
    K: int,
    mode: str = "one_step",  # "one_step" or "rollout"
    seed: int = 0,
) -> np.ndarray:
    """
    y_full: full original series of length T (including first K used as conditioning).
    samples: (m, 2+K) = [alpha, beta1..betaK, sigma]
    Returns: y_rep matrix of shape (m, T-K) for times i=K..T-1 (0-based),
             corresponding to x_{K+1: T} in 1-based indexing.
    """
    rng = np.random.default_rng(seed)
    y_full = np.asarray(y_full, dtype=float).reshape(-1)
    T = y_full.shape[0]
    m = samples.shape[0]

    alpha = samples[:, 0]
    betas = samples[:, 1:1+K]
    sigma = samples[:, 1+K]
    if np.any(sigma <= 0):
        # corner posterior should have sigma > 0, but be safe
        raise ValueError("Posterior samples contain non-positive sigma.")

    out_len = T - K
    y_rep = np.zeros((m, out_len), dtype=float)

    # initial conditioning values are always the observed first K
    y0 = y_full[:K].copy()

    if mode == "one_step":
        # Use observed lags for each step (no simulated feedback)
        for t in range(K, T):
            # lag vector: [y_{t-1},...,y_{t-K}]
            lags = np.array([y_full[t-s] for s in range(1, K+1)], dtype=float)  # (K,)
            mean_t = alpha + (betas @ lags)  # (m,)
            y_rep[:, t-K] = rng.normal(loc=mean_t, scale=sigma)
        return y_rep

    if mode == "rollout":
        # Simulate sequentially: future lags come from previous simulated values
        for k in range(m):
            path = np.zeros(T, dtype=float)
            path[:K] = y0
            a = alpha[k]
            b = betas[k]
            s = sigma[k]
            for t in range(K, T):
                lags = np.array([path[t-s_] for s_ in range(1, K+1)], dtype=float)
                mu = a + float(np.dot(b, lags))
                path[t] = rng.normal(loc=mu, scale=s)
            y_rep[k, :] = path[K:]
        return y_rep

    raise ValueError(f"Unknown mode='{mode}'.")


def _summarise_bands(y_rep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns mean, q025, q975 pointwise over draws.
    y_rep shape: (m, n_time)
    """
    mean = np.mean(y_rep, axis=0)
    lo = np.quantile(y_rep, 0.025, axis=0)
    hi = np.quantile(y_rep, 0.975, axis=0)
    return mean, lo, hi


def summarise_samples(samples, K, name):
    alpha = samples[:, 0]
    betas = samples[:, 1:1 + K]
    sigma = samples[:, 1 + K]

    print(f"\n{name}")
    print("alpha:", np.mean(alpha), np.std(alpha))
    for j in range(K):
        print(f"beta{j + 1}:", np.mean(betas[:, j]), np.std(betas[:, j]))
    print("sigma:", np.mean(sigma), np.std(sigma))


@hydra.main(version_base="1.1", config_path="../../configs/paper/ksd_calculation/real/", config_name="ark_posteriordb")
def plot_posterior_predictive(cfg: DictConfig) -> None:
    prefix = cfg.playground.get("output_prefix", "ark_param")
    plot_config_path = os.path.join(get_original_cwd(), "configs/plots/overleaf_plots_settings.yaml")
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    plot_cfg = load_plot_config(plot_config_path)

    posterior_draws_corner_prior_path = "/Users/arinaodv/Desktop/folder/study_phd/code/posteriordb/posterior_database/reference_posteriors/draws/draws/arK-arK-corner.json"
    model = instantiate(cfg.model, data_config=cfg.data)

    # observations
    y_obs = model.observations.reshape(-1)          # length T-K
    y_full = model.x.reshape(-1)                    # length T (you already have this)
    K = int(model.K)

    # ---------- CORNER posterior samples ----------
    warmup = 1000
    chains = _load_corner_draws_json(posterior_draws_corner_prior_path)
    corner_samples = _stack_corner_chains(
        chains=chains,
        K=K,
        warmup=warmup,
        max_draws=None,
    )

    # ---------- REFERENCE posterior samples ----------
    ref_samples = model.posterior_samples_init
    if ref_samples.ndim != 2 or ref_samples.shape[1] != (K + 2):
        raise ValueError(f"reference posterior_samples_init has shape {ref_samples.shape}, expected (*, {K+2}).")

    summarise_samples(ref_samples, K, "reference posterior")
    summarise_samples(corner_samples, K, "corner posterior")

    # ---------- POSTERIOR PREDICTIVE ----------
    mode = "one_step"
    seed = 27

    y_rep_corner = _ar_posterior_predictive(
        y_full=y_full,
        samples=corner_samples,
        K=K,
        mode=mode,
        seed=seed,
    )
    corner_mean, corner_lo, corner_hi = _summarise_bands(y_rep_corner)

    y_rep_ref = _ar_posterior_predictive(
        y_full=y_full,
        samples=ref_samples,
        K=K,
        mode=mode,
        seed=seed,
    )
    ref_mean, ref_lo, ref_hi = _summarise_bands(y_rep_ref)

    plot_posterior_predictive_bands_compare(
        plot_cfg=plot_cfg,
        output_dir=output_dir,
        prefix=prefix,
        y_obs=y_obs,
        ref_mean=ref_mean,
        ref_lo=ref_lo,
        ref_hi=ref_hi,
        corner_mean=corner_mean,
        corner_lo=corner_lo,
        corner_hi=corner_hi,
        filename="ark-posterior-predictive.pdf",
    )


if __name__ == "__main__":
    # main()
    # plot_posterior_predictive()
    compare_complexities()
