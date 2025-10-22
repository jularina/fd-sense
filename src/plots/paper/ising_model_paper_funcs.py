import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.cm as cmx
from typing import Dict, Tuple, List, Optional, Sequence
from scipy.special import logsumexp

from utils.distributions import DISTRIBUTION_MAP as _DIST_MAP_EXTPROJ


def _deep_get(cfg, path, default=None):
    """Safe nested getter that works with dicts or OmegaConf/objects."""
    if cfg is None:
        return default
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

def _auto_x_range_from_ref(family: str, ref_params: Dict[str, float]) -> Tuple[float, float]:
    low = family.strip().lower()

    # Gaussian / Normal
    if low in ("gaussian", "normal"):
        mu = float(ref_params["mu"])
        s  = float(ref_params["sigma"])
        return (mu - 4.0 * s, mu + 4.0 * s)

    if low in ("halfcauchy"):
        gamma = float(ref_params.get("gamma", ref_params.get("scale", 1.0)))
        return (0.0, 10.0 * gamma)

    # LogNormal with parameters mu_log, sigma_log (accept "sigma-log" typo too)
    if low == "lognormal":
        mu_log   = float(ref_params["mu_log"])
        sigma_log = float(ref_params.get("sigma_log", ref_params.get("sigma-log")))
        lo = np.exp(mu_log - 5.0 * sigma_log)
        hi = np.exp(mu_log + 5.0 * sigma_log)
        # Ensure strictly positive lower bound
        return (max(1e-8, lo), hi)

    # Gamma with shape alpha and scale theta; support x > 0
    if low == "gamma":
        alpha = float(ref_params["alpha"])
        theta = float(ref_params["theta"])
        mean = alpha * theta
        sd   = np.sqrt(alpha) * theta
        return (0.0, max(1e-8, mean + 6.0 * sd))

    # Fallback
    return (-5.0, 5.0)

def _linspace_pad(xmin, xmax, n=1200, pad=0.05):
    if not np.isfinite([xmin, xmax]).all():
        xmin, xmax = -5.0, 5.0
    if xmin == xmax:
        xmin -= 1.0; xmax += 1.0
    span = xmax - xmin
    return np.linspace(xmin - pad*span, xmax + pad*span, n)

def _make_pdf(family: str, params: Dict[str, float]):
    fam = family.strip()
    Dist = _DIST_MAP_EXTPROJ[fam]
    dist = Dist(**params)
    def f(x):
        y = dist.pdf(x)
        return np.asarray(y).squeeze()
    return f

def _parse_family_from_target(target: str) -> str:
    return target.split(".")[-1]

def _get_base_prior_spec(cfg) -> Tuple[str, Dict[str, float]]:
    d = _deep_get(cfg, f"data.base_prior", None)
    target = d.get("_target_", "")
    family = _parse_family_from_target(target)
    params = {k: float(v) for k, v in d.items() if k != "_target_"}
    return family, params

def _sample_param_sets(ranges: Dict[str, List[float]], n: int, rng: np.random.Generator):
    keys = list(ranges.keys())
    lows = np.array([ranges[k][0] for k in keys], dtype=float)
    highs = np.array([ranges[k][1] for k in keys], dtype=float)
    out = []
    for _ in range(n):
        u = rng.random(len(keys))
        vals = lows + (highs - lows) * u
        out.append({k: float(v) for k, v in zip(keys, vals)})
    return out

def plot_theta_prior(
    theta: str,
    worst_corner: Dict,
    cfg,
    plot_cfg,
    output_dir: str,
    sample_n: int = 30,
    seed: int = 123,
    filename: str = None,
    x: tuple = None,
):
    _apply_plot_rc(plot_cfg)
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    col_ref = "black"
    col_red = "red"
    alpha_cloud = 0.18
    palette_full = list(plot_cfg.plot.color_palette.colors)

    def _cloud_colors(n: int, skip_first: int = 2) -> List[str]:
        base = palette_full[skip_first:] if skip_first < len(palette_full) else palette_full
        reps = int(np.ceil(n / len(base)))
        return (base * reps)[:n]

    fam_ref, ref_params = _get_base_prior_spec(cfg)
    ranges = cfg.ksd.optimize.prior.Gamma["parameters_box_range"]["ranges"]
    fam_ms = worst_corner["family"]
    ms_params = worst_corner["params"]

    X = np.linspace(0, 15, 200)

    pdf_ref = _make_pdf(fam_ref, ref_params)
    pdf_ms  = _make_pdf(fam_ms,  ms_params)

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi
    )
    ax = fig.add_subplot(111)

    ax.plot(X, pdf_ref(X), linestyle="--", color=col_ref, linewidth=1.2, label=plot_cfg.plot.param_latex_names["baseprior"])
    for (p, c) in zip(_sample_param_sets(ranges, sample_n, rng), _cloud_colors(sample_n, 2)):
        pdf_c = _make_pdf(fam_ms, p)
        ax.plot(X, pdf_c(X), linewidth=0.9, alpha=alpha_cloud, color=c)
    ax.plot(X, pdf_ms(X), color=col_red, linewidth=1.8, label=r"$\Pi$ (" + plot_cfg.plot.param_latex_names["argoptimisationProblemParam"] +  " )")

    ax.set_ylabel(plot_cfg.plot.param_latex_names.priorsimple)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.6))

    if getattr(plot_cfg.plot.figure, "tight_layout", True):
        plt.tight_layout()

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_lr_vs_ksd(
    lr_grid,
    plot_cfg,
    output_dir: str,
    filename: str = "sbi_experiment_turin_lr_scan.pdf",
    xlabel: str = None,
    title: str = None,
):
    _apply_plot_rc(plot_cfg)

    pairs = []
    for item in lr_grid:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            lr, ksd = float(item[0]), float(item[1])
            pairs.append((lr, ksd))
        elif isinstance(item, dict):
            lr_key = "lr" if "lr" in item else ("learning_rate" if "learning_rate" in item else None)
            ksd_key = "ksd" if "ksd" in item else ("value" if "value" in item else None)
            if lr_key is None or ksd_key is None:
                continue
            pairs.append((float(item[lr_key]), float(item[ksd_key])))

    if not pairs:
        raise ValueError("plot_lr_vs_ksd: empty or unrecognized lr_grid format.")

    # sort by lr
    pairs.sort(key=lambda t: t[0])
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)

    # ---- figure ----
    W = plot_cfg.plot.figure.size.width
    H = plot_cfg.plot.figure.size.height
    fig, ax = plt.subplots(1, 1, figsize=(W, H), dpi=plot_cfg.plot.figure.dpi)

    # line + markers
    ax.plot(xs, ys, marker="o", linewidth=1.5, color=plot_cfg.plot.color_palette.colors[1], markersize=4)

    # aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # remove y-axis ticks and labels completely
    # ax.set_yticks([])
    ax.set_ylabel(plot_cfg.plot.param_latex_names["estimatedKSDposteriorsShort"])
    # ax.tick_params(axis="y", left=False, right=False, labelleft=False)

    # labels
    if xlabel is None:
        xlabel = r"lr"
    ax.set_xlabel(xlabel)

    xmin, xmax = float(xs.min()), float(xs.max())
    if xmin > 0 and (xmax / max(xmin, 1e-20) > 1e3):
        ax.set_xscale("log")

    if title:
        ax.set_title(title)

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_lr_vs_ksd_multi(
    lr_grids,
    ds,
    plot_cfg,
    output_dir: str,
    filename: str = "ising_experiment_multi_d_lr_vs_ksd.pdf",
    xlabel: str = None,
    legend: bool = True,
    ylbl: str = "estimatedKSDposteriorsShort",
):
    _apply_plot_rc(plot_cfg)

    W = plot_cfg.plot.figure.size.width
    H = plot_cfg.plot.figure.size.height
    fig, ax = plt.subplots(1, 1, figsize=(W, H), dpi=plot_cfg.plot.figure.dpi)

    colors = plot_cfg.plot.color_palette.colors
    for i, (grid, d) in enumerate(zip(lr_grids, ds)):
        xs = np.array(grid[:, 0], dtype=float)
        ys = np.array(grid[:, 1], dtype=float)
        idx = np.argsort(xs)
        xs, ys = xs[idx], ys[idx]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.5,
                color=colors[i % len(colors)], label=fr"$d={d}$")

        max_idx = np.argmax(ys)
        ax.plot(xs[max_idx], ys[max_idx], marker="*", color="red", markersize=6, zorder=5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(plot_cfg.plot.param_latex_names[ylbl])
    ax.set_xlabel(xlabel if xlabel is not None else r"lr")

    xmin = min(grid[:, 0].min() for grid in lr_grids)
    xmax = max(grid[:, 0].max() for grid in lr_grids)
    if xmin > 0 and (xmax / max(xmin, 1e-20) > 1e3):
        ax.set_xscale("log")

    if legend:
        ax.legend()

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_sdp_densities_only(
    basis_function,
    psi_sdp_list: list[np.ndarray],
    radius_labels: list[float],
    ksd_estimates: list[float],
    prior_distribution,
    plot_cfg,
    output_dir: str,
    domain: tuple = (-2, 5),
    resolution: int = 200,
    filename = "toy_gaussian_model_nonparametric_optimisation_densities.pdf",
    ylbl: str = "estimatedKSDposteriorsShort",
) -> None:
    """
    Plot true prior density and normalized SDP densities (no samples).
    Matches styling (legend, colors, fonts, LaTeX) used in plot_sdp_comparisons_multiple_radii.
    Saves a single-panel figure to PDF.
    """
    os.makedirs(output_dir, exist_ok=True)

    _apply_plot_rc(plot_cfg)

    # Figure
    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Grid + features
    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])
    Phi_x = basis_function.evaluate(x)  # (resolution, D)

    # True prior density
    prior_density = prior_distribution.pdf(x).flatten()

    # Colors (cycle through palette for each radius curve)
    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]  # fallback

    # Names/labels map
    names = plot_cfg.plot.param_latex_names
    ksd_label = names.get(ylbl)
    xlabel = names.get("theta")
    ylabel = names.get("nonparametric_prior", "Density")
    true_label = names.get("baseprior", "True Prior Density")
    approx_sym = r"$\approx$"
    geq_sym = r"$\geq$"

    # Plot SDP densities (normalized on grid)
    for i, (psi, r_label, ksd) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
        f = (Phi_x @ psi).flatten()
        logZ = logsumexp(f) + np.log(dx)
        p_hat = np.exp(f - logZ)
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label:.2f} ({ksd:.1f})"
        ax.plot(
            x.flatten(),
            p_hat,
            label=label,
            linewidth=1.5,
            color=color,
        )

    # Plot true prior density (on top/last or first as you prefer)
    ax.plot(
        x.flatten(),
        prior_density,
        label=true_label,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    centers = np.array(basis_function.centers).flatten()
    # ax.scatter(centers, np.zeros_like(centers), color="grey", marker="o", s=10, alpha=0.6, label=None)
    y_min = -0.08 * max(prior_density)  # 5% below axis
    ax.scatter(
        centers,
        np.full_like(centers, y_min),
        color="grey",
        marker="o",
        s=10,
        alpha=0.8,
        label=None,
    )

    # Axes labels & styling
    ax.set_xlabel(xlabel, fontsize=plot_cfg.plot.font["size"])
    ax.set_ylabel(ylabel, fontsize=plot_cfg.plot.font["size"])
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend styling identical to your other function
    leg = ax.legend(
        title=ksd_label,
        loc="center left",
        bbox_to_anchor=(1.02, 0.6),
        frameon=False,
        labelspacing=0.3,
        handlelength=0.2,
        handletextpad=0.3,
        borderpad=0.1,
    )
    for t in leg.get_texts():
        t.set_wrap(True)
    leg.get_title().set_ha("right")
    leg._legend_box.align = "right"

    plt.setp(leg.get_texts(), fontsize=plt.rcParams["font.size"]*0.8)
    plt.setp(leg.get_title(), fontsize=plt.rcParams["font.size"]*0.8)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    # Save and close
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_basis_colored_by_eigenvector(
    basis_function,
    eigenvector: np.ndarray,
    plot_cfg,
    output_dir: str,
    domain: tuple = (-3, 12),
    resolution: int = 300,
    filename: str = "basis_colored_by_eigenvector.pdf",
    cmap_name: str = "coolwarm",
    linewidth_base: float = 1.5,
    linewidth_scale: float = 1.0,
    show_colorbar: bool = True,
) -> None:
    """
    Plot each basis function φ_k(x) colored by the corresponding eigenvector entry (length K).

    Args:
        basis_function: object with .evaluate(x)->(N,K) and .centers (optional).
        eigenvector: shape (K,), entries used to color each basis curve.
        plot_cfg: your existing plot config (used by _apply_plot_rc).
        output_dir: directory to save the PDF.
        domain: (xmin, xmax) for plotting grid.
        resolution: number of grid points.
        filename: output PDF filename.
        cmap_name: matplotlib colormap name (use a diverging cmap for +/- weights).
        linewidth_base: base linewidth for curves.
        linewidth_scale: additional scaling by |eigenvector| (0 disables scaling).
        show_colorbar: whether to draw a colorbar for eigenvector values.

    Saves:
        A single-panel PDF showing all K basis curves colored by eigenvector entries.
    """
    from matplotlib import cm, colors
    os.makedirs(output_dir, exist_ok=True)

    # Apply your global plot styling (must exist in your code base)
    _apply_plot_rc(plot_cfg)

    # Prepare grid and evaluate basis
    x = np.linspace(domain[0], domain[1], resolution)[:, None]  # (N,1)
    Phi_x = basis_function.evaluate(x)  # (N, K)
    if Phi_x.ndim == 3 and Phi_x.shape[1] == 1:
        Phi_x = Phi_x[:, 0, :]
    N, K = Phi_x.shape

    eigenvector = np.asarray(eigenvector).reshape(-1)
    if eigenvector.shape[0] != K:
        raise ValueError(f"eigenvector length {eigenvector.shape[0]} does not match number of basis functions K={K}.")

    # Figure
    fig, ax = plt.subplots(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Build colormap normalizer centered at 0 for +/- weights
    vmax = np.max(np.abs(eigenvector)) if np.any(eigenvector) else 1.0
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    # (Optional) draw centers along the baseline for reference
    centers = np.array(getattr(basis_function, "centers", [])).flatten()
    # Estimate a baseline a bit below visible y-range using a rough amplitude proxy
    # (use the max absolute value over all basis to scale baseline offset)
    amp_proxy = float(np.max(np.abs(Phi_x))) if Phi_x.size else 1.0
    y_min = -0.08 * amp_proxy
    if centers.size > 0:
        ax.scatter(
            centers,
            np.full_like(centers, y_min),
            color="grey",
            marker="o",
            s=10,
            alpha=0.8,
            label=None,
        )

    # Plot each basis function colored by its eigenvector entry
    for k in range(K):
        yk = Phi_x[:, k]
        w = eigenvector[k]
        color = cmap(norm(w))

        # Optional: make lines a bit thicker for larger |weight|
        lw = linewidth_base + linewidth_scale * (abs(w) / vmax if vmax > 0 else 0.0)

        ax.plot(
            x.flatten(),
            yk,
            color=color,
            linewidth=lw,
            alpha=0.95,
        )

    # Labels & axes styling
    names = plot_cfg.plot.param_latex_names
    xlabel = names.get("theta", r"$\theta$")
    ylabel = names.get("basis_functions", "Basis Functions")
    ax.set_xlabel(xlabel, fontsize=plot_cfg.plot.font["size"])
    ax.set_ylabel(ylabel, fontsize=plot_cfg.plot.font["size"])
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colorbar to explain mapping eigenvector value -> color
    if show_colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(names.get("eigenvector_value", "Eigenvector Entry"), rotation=90)

    # Tight layout if configured
    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    # Save and close
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)