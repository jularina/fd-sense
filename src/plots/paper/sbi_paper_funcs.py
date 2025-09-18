import os
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from src.utils.distributions import DISTRIBUTION_MAP

# ---- util helpers (use the ones you pasted; kept here for clarity) ----
def _deep_get(cfg, path, default=None):
    cur = cfg
    for key in path.split('.'):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur

def _apply_plot_rc(plot_cfg):
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{type1cm}",
    })

def _save_fig(fig, output_dir: str, filename: str, plot_cfg):
    os.makedirs(output_dir, exist_ok=True)
    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format=filename.split(".")[-1], bbox_inches="tight")
    plt.close(fig)

# ============== main plotting function ==============

def plot_turin_four_theta_priors(
    largest_sens: Dict,     # dict like {'theta_1': {'family': 'LogNormal', 'params': {...}}, ...}
    cfg,                    # hydra/omegaconf config containing data.base_prior.distributions.theta_i.{low,high}
    plot_cfg,               # plotting config (rc + sizes)
    output_dir: str,
    filename: str = "sbi_experiment_turin_prior_four_panel.pdf",
    n_points: int = 600
):
    """
    Make a 1x4 figure: for theta_1..theta_4 plot base Uniform prior (from cfg)
    and the 'largest sensitivity' candidate prior (from largest_sens).
    """
    _apply_plot_rc(plot_cfg)

    # ---- internal helpers ----
    def _base_uniform_params(theta_name: str) -> Tuple[float, float]:
        base_path = f"data.base_prior.distributions.{theta_name}"
        low = float(_deep_get(cfg, base_path + ".low"))
        high = float(_deep_get(cfg, base_path + ".high"))
        return low, high

    def _make_dist(family: str, params: Dict):
        cls = DISTRIBUTION_MAP[family]
        return cls(**params)

    def _nice_x_range_uniform(low: float, high: float, candidate_family: str, candidate_params: Dict) -> Tuple[float, float]:
        """Build an x-range that covers both the base Uniform and the bulk of candidate."""
        # start from an expanded uniform window
        xmin = max(1e-20, low * 0.5) if low > 0 else min(low - 0.1 * (high - low), low - 0.1)
        xmax = high * 2.0
        # expand based on candidate scale
        if candidate_family == "Gamma":
            # mean ~ alpha*theta; std ~ sqrt(alpha)*theta
            a = float(candidate_params["alpha"]); th = float(candidate_params["theta"])
            mean = a * th; std = (a ** 0.5) * th
            xmin = min(xmin, max(1e-20, mean - 5 * std))
            xmax = max(xmax, mean + 5 * std)
        elif candidate_family == "LogNormal":
            mu = float(candidate_params["mu_log"]); sig = float(candidate_params["sigma_log"])
            # cover exp(mu ± 5σ)
            xmin = min(xmin, np.exp(mu - 5 * sig))
            xmax = max(xmax, np.exp(mu + 5 * sig))
        elif candidate_family in ("Uniform", "HalfCauchy", "Cauchy", "Gaussian", "Laplace"):
            # fallback: mild padding
            pad = 0.1 * (xmax - xmin)
            xmin = min(xmin, low - pad)
            xmax = max(xmax, high + pad)
        return max(1e-20, xmin), xmax

    def _x_grid(xmin, xmax):
        # log-aware spacing for wide positive ranges, else linear
        if xmin > 0 and xmax / max(xmin, 1e-20) > 1e3:
            xs = np.geomspace(xmin, xmax, n_points)
        else:
            xs = np.linspace(xmin, xmax, n_points)
        return xs

    # ---- build figure ----
    W = plot_cfg.plot.figure.size.width
    H = plot_cfg.plot.figure.size.height
    fig, axes = plt.subplots(1, 4, figsize=(W, H), dpi=plot_cfg.plot.figure.dpi, sharey=False)
    col_ref = "black"
    col_cand = "red"

    theta_labels = {
        "theta_1": r"$\theta_1$",
        "theta_2": r"$\theta_2$",
        "theta_3": r"$\theta_3$",
        "theta_4": r"$\theta_4$",
    }

    for ax, tname in zip(axes, ["theta_1", "theta_2", "theta_3", "theta_4"]):
        # base prior (Uniform from cfg)
        low, high = _base_uniform_params(tname)
        base = _make_dist("Uniform", {"low": low, "high": high})
        # candidate (largest sensitivity)
        cand_info = largest_sens[tname]
        cand_fam = cand_info["family"]
        cand_params = cand_info["params"]
        cand = _make_dist(cand_fam, cand_params)

        # x-range covering both
        xmin, xmax = _nice_x_range_uniform(low, high, cand_fam, cand_params)
        x = _x_grid(xmin, xmax)

        # pdfs
        y_base = base.pdf(x)
        y_cand = cand.pdf(x)

        # plot
        ax.plot(x, y_base, linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")
        ax.plot(x, y_cand, color=col_cand, linewidth=1.8, label=r"$\Pi$ (sup$_{\gamma \in C_\gamma} L_m(\gamma)$)")

        # aesthetics
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(theta_labels[tname])
        if tname == "theta_1":
            ylabel = plot_cfg.plot.param_latex_names["nonparametric_prior"]
            ax.set_ylabel(ylabel)

        # keep readable ticks on log-like spans
        if xmin > 0 and xmax / max(xmin, 1e-20) > 1e3:
            ax.set_xscale("log")

    # one shared legend
    handles = [
        plt.Line2D([], [], color=col_ref, linestyle="--", linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$"),
        plt.Line2D([], [], color=col_cand, linestyle="-", linewidth=1.8, label=r"$\Pi$ (sup$_{\gamma \in C_\gamma} L_m(\gamma)$)"),
    ]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.09))

    _save_fig(fig, output_dir, f"{filename}", plot_cfg)