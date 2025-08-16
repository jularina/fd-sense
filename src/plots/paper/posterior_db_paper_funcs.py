import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.cm as cmx
from typing import Dict, Tuple, List, Optional, Sequence


def _apply_plot_rc(plot_cfg):
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{type1cm}",
    })

def _new_fig_ax(plot_cfg):
    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    return fig, ax

def _save_fig(fig, output_dir: str, filename: str, plot_cfg):
    os.makedirs(output_dir, exist_ok=True)
    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format=filename.split(".")[-1], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def _palette(plot_cfg, n: int) -> List[str]:
    base = list(plot_cfg.plot.color_palette.colors)
    if n <= len(base):
        return base[:n]
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]

def _latex_name(plot_cfg, key: str, default: str) -> str:
    try:
        return getattr(plot_cfg.plot.param_latex_names, key)
    except Exception:
        return default

def _gaussian_pdf(x, mu, sigma):
    x = np.asarray(x, float)
    s2 = sigma * sigma
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * (x - mu) ** 2 / s2)

def _lognormal_pdf(x, mu_log, sigma_log):
    x = np.asarray(x, float)
    pdf = np.zeros_like(x)
    pos = x > 0
    z = (np.log(x[pos]) - mu_log) / sigma_log
    pdf[pos] = (1.0 / (x[pos] * sigma_log * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)
    return pdf

def _get_base_gaussian(cfg, name: str) -> Tuple[float, float]:
    dist = cfg.data.base_prior.distributions[name]
    return float(dist.mu), float(dist.sigma)

def _extract_worst(rows: List[Dict]) -> Dict:
    return max(rows, key=lambda r: float(r["value"]))

def _extract_params(d: Dict, key: str) -> Dict:
    """Return dict of params for given key (beta1/beta3) if present."""
    if not isinstance(d, dict):
        return {}
    return d.get(key, {}).get("params", {})

def plot_ar_time_series(
    y: np.ndarray,
    plot_cfg,
    output_dir: str,
    filename: str = "ar_pred_ribbon.pdf",
):
    """
    y: (T,)
    y_rep: (n_draws, T) or None; if None, only observed series is plotted
    """
    _apply_plot_rc(plot_cfg)
    fig, ax = _new_fig_ax(plot_cfg)

    T = len(y)
    t = np.arange(1, T + 1)

    # lines
    ax.plot(t, y, label=_latex_name(plot_cfg, "observed", "observed"), linewidth=1.5)
    ax.set_xlabel(_latex_name(plot_cfg, "time_t", r"time $t$"))
    ax.set_ylabel(_latex_name(plot_cfg, "y_t", r"$y_t$"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_fig(fig, output_dir, filename, plot_cfg)

def plot_three_panel_priors(
    rows_A: List[Dict],
    rows_B: List[Dict],
    rows_C: List[Dict],
    cfg,
    plot_cfg,
    output_dir: str,
    prefix: str = "ark_param",
    # β1 is Gaussian now:
    beta1_mu_range=(0.0, 5.0),
    beta1_sigma_range=(2.0, 5.0),
    # β3 (Gaussian):
    beta3_mu_range=(0.0, 5.0),
    beta3_sigma_range=(2.0, 5.0),
    sample_n_beta1: int = 30,
    sample_n_beta3: int = 30,
    seed: int = 123,
    filename: str = None,
):
    _apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"{prefix}_three_panel_priors.pdf"

    # Base (init) priors
    base_b1_mu, base_b1_sigma = _get_base_gaussian(cfg, "beta1")  # Gaussian (now)
    base_b3_mu, base_b3_sigma = _get_base_gaussian(cfg, "beta3")  # Gaussian

    # Worst corners
    worst_A = _extract_worst(rows_A)
    worst_B = _extract_worst(rows_B)
    worst_C = _extract_worst(rows_C)

    b1A = _extract_params(worst_A["prior_corner"], "beta1")  # expect Gaussian now
    b3A = _extract_params(worst_A["prior_corner"], "beta3")
    b1B = _extract_params(worst_B["prior_corner"], "beta1")
    b3C = _extract_params(worst_C["prior_corner"], "beta3")

    palette = _palette(plot_cfg, 3)
    col_ref = "black"
    col_worst_b1 = palette[0]
    col_worst_b3 = palette[1]
    col_grey = (0.65, 0.65, 0.65)
    col_red  = "red"

    # Figure & gridspec
    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi
    )
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 1.0], hspace=0.3, wspace=0.25)
    ax_top = fig.add_subplot(gs[0, :])
    ax_bL  = fig.add_subplot(gs[1, 0])
    ax_bR  = fig.add_subplot(gs[1, 1])

    # ---------------- TOP: reference vs worst β1 & β3 ----------------
    xs = [
        base_b1_mu - 4*base_b1_sigma, base_b1_mu + 4*base_b1_sigma,
        base_b3_mu - 4*base_b3_sigma, base_b3_mu + 4*base_b3_sigma
    ]
    # β1 worst (prefer Gaussian, fallback to LogNormal if legacy rows)
    if "mu" in b1A and "sigma" in b1A:
        xs += [float(b1A["mu"]) - 4*float(b1A["sigma"]), float(b1A["mu"]) + 4*float(b1A["sigma"])]
    elif "mu_log" in b1A and "sigma_log" in b1A:  # legacy support
        m, s = float(b1A["mu_log"]), float(b1A["sigma_log"])
        xs += [0.0, np.exp(max(m - 3*s, -12)), np.exp(min(m + 3*s, 12))]
    # β3 worst (Gaussian)
    if "mu" in b3A and "sigma" in b3A:
        xs += [float(b3A["mu"]) - 4*float(b3A["sigma"]), float(b3A["mu"]) + 4*float(b3A["sigma"])]

    xmin, xmax = min(xs), max(xs)
    x = np.linspace(xmin - 0.05*(xmax-xmin), xmax + 0.05*(xmax-xmin), 900)

    # Reference (init) — show β1's base Gaussian
    l_base, = ax_top.plot(
        x, _gaussian_pdf(x, base_b1_mu, base_b1_sigma),
        color=col_ref, linestyle="--", linewidth=1.2,
        label=_latex_name(plot_cfg, "baseprior", "baseprior")
    )

    # Worst β1 (top)
    l_wb1 = None
    if "mu" in b1A and "sigma" in b1A:
        l_wb1, = ax_top.plot(
            x, _gaussian_pdf(x, float(b1A["mu"]), float(b1A["sigma"])),
            color=col_worst_b1, linewidth=1.8,
            label=r"$\beta_1$"
        )
    elif "mu_log" in b1A and "sigma_log" in b1A:
        l_wb1, = ax_top.plot(
            x, _lognormal_pdf(x, float(b1A["mu_log"]), float(b1A["sigma_log"])),
            color=col_worst_b1, linewidth=1.8,
            label=r"$\beta_1$"
        )

    # Worst β3 (top)
    l_wb3 = None
    if "mu" in b3A and "sigma" in b3A:
        l_wb3, = ax_top.plot(
            x, _gaussian_pdf(x, float(b3A["mu"]), float(b3A["sigma"])),
            color=col_worst_b3, linewidth=1.8,
            label=r"$\beta_3$"
        )

    ax_top.set_ylabel(_latex_name(plot_cfg, "nonparametric_prior", "prior"))
    ax_top.spines["top"].set_visible(False); ax_top.spines["right"].set_visible(False)

    # ---------------- BOTTOM-LEFT: β1 (Gaussian now) ----------------
    mu_vals_b1 = np.linspace(beta1_mu_range[0], beta1_mu_range[1], 20)
    sg_vals_b1 = np.linspace(beta1_sigma_range[0], beta1_sigma_range[1], 4)
    grid_b1 = np.array([(m, s) for m in mu_vals_b1 for s in sg_vals_b1])

    rng = np.random.default_rng(seed)
    n_b1 = min(sample_n_beta1, len(grid_b1))
    sample_b1 = grid_b1[rng.choice(len(grid_b1), size=n_b1, replace=False)]

    # x-range for Gaussians: cover base + samples
    xL_min = min(np.min(sample_b1[:, 0] - 3.0*sample_b1[:, 1]), base_b1_mu - 4*base_b1_sigma)
    xL_max = max(np.max(sample_b1[:, 0] + 3.0*sample_b1[:, 1]), base_b1_mu + 4*base_b1_sigma)
    xL = np.linspace(xL_min, xL_max, 600)

    # init β1
    ax_bL.plot(xL, _gaussian_pdf(xL, base_b1_mu, base_b1_sigma),
               color=col_ref, linestyle="--", linewidth=1.2)

    # grey cloud
    for m, s in sample_b1:
        ax_bL.plot(xL, _gaussian_pdf(xL, float(m), float(s)),
                   color=col_grey, alpha=0.6, linewidth=0.9)

    # worst from rows_B (no label; legend handled globally)
    if "mu" in b1B and "sigma" in b1B:
        ax_bL.plot(xL, _gaussian_pdf(xL, float(b1B["mu"]), float(b1B["sigma"])),
                   color=col_red, linewidth=1.9)
    elif "mu_log" in b1B and "sigma_log" in b1B:
        xL_pos = np.linspace(
            max(1e-6, np.exp(float(b1B["mu_log"]) - 6*float(b1B["sigma_log"]))),
            np.exp(float(b1B["mu_log"]) + 6*float(b1B["sigma_log"])), 600
        )
        ax_bL.plot(xL_pos, _lognormal_pdf(xL_pos, float(b1B["mu_log"]), float(b1B["sigma_log"])),
                   color=col_red, linewidth=1.9)

    ax_bL.set_xlabel(r"$\beta_1$")
    ax_bL.set_ylabel(_latex_name(plot_cfg, "prior_beta1", "prior_beta1"))
    ax_bL.spines["top"].set_visible(False); ax_bL.spines["right"].set_visible(False)
    # (no ax_bL.legend)

    # ---------------- BOTTOM-RIGHT: β3 (Gaussian) ----------------
    mu_vals3 = np.linspace(beta3_mu_range[0], beta3_mu_range[1], 6)
    sg_vals3 = np.linspace(beta3_sigma_range[0], beta3_sigma_range[1], 4)
    grid_b3 = np.array([(m, s) for m in mu_vals3 for s in sg_vals3])

    n_b3 = min(sample_n_beta3, len(grid_b3))
    sample_b3 = grid_b3[rng.choice(len(grid_b3), size=n_b3, replace=False)]

    xR_min = min(np.min(sample_b3[:, 0] - 3.0*sample_b3[:, 1]), base_b3_mu - 4*base_b3_sigma)
    xR_max = max(np.max(sample_b3[:, 0] + 3.0*sample_b3[:, 1]), base_b3_mu + 4*base_b3_sigma)
    xR = np.linspace(xR_min, xR_max, 600)

    ax_bR.plot(xR, _gaussian_pdf(xR, base_b3_mu, base_b3_sigma),
               color=col_ref, linestyle="--", linewidth=1.2)

    for m, s in sample_b3:
        ax_bR.plot(xR, _gaussian_pdf(xR, float(m), float(s)),
                   color=col_grey, alpha=0.6, linewidth=0.9)

    # worst from rows_C (no label; legend handled globally)
    if "mu" in b3C and "sigma" in b3C:
        ax_bR.plot(xR, _gaussian_pdf(xR, float(b3C["mu"]), float(b3C["sigma"])),
                   color=col_red, linewidth=1.9)

    ax_bR.set_xlabel(r"$\beta_3$")
    ax_bR.set_ylabel(_latex_name(plot_cfg, "prior_beta3", "prior_beta3"))
    ax_bR.spines["top"].set_visible(False); ax_bR.spines["right"].set_visible(False)
    # (no ax_bR.legend)

    # ---------------- SINGLE, FIGURE-LEVEL LEGEND (right side) ----------------
    # Use the three canonical handles from the TOP plot:
    handles = []
    labels  = []
    if l_base is not None:
        handles.append(l_base); labels.append(l_base.get_label())
    if l_wb1 is not None:
        handles.append(l_wb1); labels.append(l_wb1.get_label())
    if l_wb3 is not None:
        handles.append(l_wb3); labels.append(l_wb3.get_label())

    # make room on the right for legend
    # fig.subplots_adjust(right=0.82)
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.00, 0.5),
        frameon=False
    )

    # Save
    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()
    _save_fig(fig, output_dir, filename, plot_cfg)


