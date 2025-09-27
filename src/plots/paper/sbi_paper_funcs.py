import os
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from src.utils.distributions import DISTRIBUTION_MAP


SCALE_MAP = {
    "theta_1": 1e9,
    "theta_2": 1e9,
    "theta_3": 1e-7,
    "theta_4": 1e10,
}

def _base_uniform_params_original(cfg, theta_name: str) -> Tuple[float, float]:
    """Read (possibly scaled) low/high from cfg and divide by s to get original-scale bounds."""
    base_path = f"data.candidate_prior.distributions.{theta_name}"
    low_s  = float(_deep_get(cfg, base_path + ".low"))
    high_s = float(_deep_get(cfg, base_path + ".high"))
    s = SCALE_MAP[theta_name]
    return (low_s / s, high_s / s)


def _candidate_params_original(theta_name: str, family: str, params: Dict) -> Tuple[str, Dict]:
    """
    Convert candidate prior params (given in model/scaled space) into original space.
    - Uniform: divide bounds by s
    - LogNormal: mu_log -= ln(s), sigma_log unchanged
    """
    s = SCALE_MAP[theta_name]
    if family == "Uniform":
        return family, {"low": params["low"] / s, "high": params["high"] / s}
    elif family == "LogNormal":
        return family, {
            "mu_log": float(params["mu_log"]) - np.log(s),
            "sigma_log": float(params["sigma_log"]),
        }
    else:
        return family, params

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
        base_path = f"data.candidate_prior.distributions.{theta_name}"
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
        "theta_1": r"$\mathrm{G}_0$",
        "theta_2": r"$T$",
        "theta_3": r"$\nu$",
        "theta_4": r"$\sigma_W^2$",
    }

    for ax, tname in zip(axes, ["theta_1", "theta_2", "theta_3", "theta_4"]):
        low, high = _base_uniform_params_original(cfg, tname)
        base = _make_dist("Uniform", {"low": low, "high": high})
        cand_info = largest_sens[tname]
        cand_fam = cand_info["family"]
        cand_params_orig_fam, cand_params_orig = _candidate_params_original(tname, cand_fam, cand_info["params"])
        cand = _make_dist(cand_params_orig_fam, cand_params_orig)

        # x-range covering both
        xmin, xmax = _nice_x_range_uniform(low, high, cand_params_orig_fam, cand_params_orig)
        x = _x_grid(xmin, xmax)

        # pdfs
        y_base = base.pdf(x)
        y_cand = cand.pdf(x)

        # plot
        ax.plot(x, y_base, linestyle="--", color=col_ref, linewidth=1.2, label=r"$\Pi_{\mathrm{ref}}$")
        ax.plot(x, y_cand, color=col_cand, linewidth=1.8, label=r"$\Pi$ (" + plot_cfg.plot.param_latex_names["argoptimisationProblemParam"] + ")")

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
        plt.Line2D([], [], color=col_cand, linestyle="-", linewidth=1.8, label=r"$\Pi$ (" + plot_cfg.plot.param_latex_names["argoptimisationProblemParam"] + ")"),
    ]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.09))

    _save_fig(fig, output_dir, f"{filename}", plot_cfg)



def plot_lr_vs_ksd(
    lr_grid,                 # list of (lr, ksd) tuples, or list of dicts with keys {'lr'|'learning_rate', 'ksd'|'value'}
    plot_cfg,                # plotting config (rc + sizes)
    output_dir: str,
    filename: str = "sbi_experiment_turin_lr_scan.pdf",
    xlabel: str = None,
    title: str = None,
):
    """
    Plot KSD vs learning rate from a grid of (lr, ksd) pairs.

    - Applies your plot rc via _apply_plot_rc(plot_cfg)
    - Uses log x-scale if lr span is wide and positive
    - Hides all y-axis ticks/labels
    - Saves via _save_fig
    """
    _apply_plot_rc(plot_cfg)

    # ---- normalize lr_grid into arrays ----
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
    ax.plot(xs, ys, marker="o", linewidth=1.8, color=plot_cfg.plot.color_palette.colors[0])

    # aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # remove y-axis ticks and labels completely
    ax.set_yticks([])
    ax.set_ylabel(plot_cfg.plot.param_latex_names["estimatedKSDposteriorsShort"])
    ax.tick_params(axis="y", left=False, right=False, labelleft=False)

    # labels
    if xlabel is None:
        xlabel = r"lr"
    ax.set_xlabel(xlabel)

    # x log-scale if lr spans orders of magnitude and positive
    xmin, xmax = float(xs.min()), float(xs.max())
    if xmin > 0 and (xmax / max(xmin, 1e-20) > 1e3):
        ax.set_xscale("log")

    if title:
        ax.set_title(title)

    _save_fig(fig, output_dir, filename, plot_cfg)


def plot_ksd_heatmap(
    data_dict,
    plot_cfg,
    output_dir: str,
    filename: str = "ksd_heatmap.pdf",
    title: str = None,  # if None -> no title
):
    basis_funcs = sorted(data_dict.keys())
    radii = sorted({float(r) for bf in data_dict for r in data_dict[bf].keys()})
    colorbar_label = plot_cfg.plot.param_latex_names["optimisationProblem"]

    # Build matrix of ksd values
    heatmap = np.full((len(radii), len(basis_funcs)), np.nan, dtype=float)
    for j, bf in enumerate(basis_funcs):
        for i, r in enumerate(radii):
            val = data_dict[bf].get(r, None)
            if val is None:
                # try key as float->str mismatch guard
                val = data_dict[bf].get(float(r), np.nan)
            heatmap[i, j] = val

    pc = getattr(plot_cfg, "plot", plot_cfg)
    param_names = getattr(pc, "param_latex_names", {"r": "r", "K": "K"})
    figsize = (plot_cfg.plot.figure.size["width"], plot_cfg.plot.figure.size["height"])
    dpi = plot_cfg.plot.figure["dpi"]
    labelsize = plot_cfg.plot.font["size"]
    tick_labelsize = getattr(pc, "tick_labelsize", 10)
    fontfamily = getattr(pc, "fontfamily", None)
    cbar_labelsize = getattr(pc, "colorbar_labelsize", labelsize)
    vmin = getattr(pc, "vmin", None)
    vmax = getattr(pc, "vmax", None)
    cmap = _resolve_cmap_from_cfg(plot_cfg)

    # Mask NaNs so they render as transparent
    Hmask = np.ma.masked_invalid(heatmap)

    rc_update = {"font.family": fontfamily} if fontfamily else {}
    with plt.rc_context(rc_update):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        cax = ax.imshow(
            Hmask,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            vmin=vmin, vmax=vmax,
        )
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(colorbar_label, fontsize=cbar_labelsize)
        cbar.ax.tick_params(labelsize=tick_labelsize)

        # Set ticks and labels
        ax.set_xticks(range(len(basis_funcs)))
        ax.set_xticklabels(basis_funcs)
        ax.set_yticks(range(len(radii)))
        ax.set_yticklabels([str(r) for r in radii])

        ax.set_xlabel(param_names.get("K", "K"), fontsize=labelsize)
        ax.set_ylabel(param_names.get("r", "r"), fontsize=labelsize)
        ax.tick_params(axis="both", which="both", labelsize=tick_labelsize)

        if title:
            ax.set_title(title, fontsize=labelsize)

        _save_fig(fig, output_dir, filename, plot_cfg)
        plt.close(fig)


def _mix(c1, c2, t):
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    return (1 - t) * c1 + t * c2

def _smooth_palette_cmap(colors, N=256):
    # Smooth, continuous gradient through the provided colors
    return LinearSegmentedColormap.from_list("cfg_palette_grad", list(colors), N=N)

def _single_hue_cmap(base_color, low_light=0.75, darken=0.35, reverse=False, N=256):
    # Low end: a tint toward white (low_light closer to 1.0 => lighter low end)
    base = np.array(to_rgb(base_color))
    white = np.array([1.0, 1.0, 1.0])
    black = np.array([0.0, 0.0, 0.0])

    low = _mix(base, white, low_light)       # lightened base for the low end
    high = _mix(base, black, darken)         # darkened base for the high end

    cmap = LinearSegmentedColormap.from_list("single_hue_boosted", [tuple(low), tuple(base), tuple(high)], N=N)
    return cmap.reversed() if reverse else cmap

def _resolve_cmap_from_cfg(plot_cfg):
    pc = getattr(plot_cfg, "plot", plot_cfg)
    palette = getattr(pc, "color_palette", None)
    colors = getattr(palette, "colors", None) if palette is not None else None
    base_color = colors[0]

    # Stronger contrast than before (less pale): lower low_light, add darken
    low_light = float(getattr(pc, "cmap_low_light", 0.75))   # 0.60–0.85 is good; 0.92 was too pale
    darken    = float(getattr(pc, "cmap_darken", 0.40))      # 0–0.6; higher = deeper high end
    reverse   = bool(getattr(pc, "cmap_reverse", False))
    N         = int(getattr(pc, "cmap_N", 256))
    return _single_hue_cmap(base_color, low_light=low_light, darken=darken, reverse=reverse, N=N)

def plot_ksd_heatmap_continuous(
    data_dict,
    plot_cfg,
    output_dir: str,
    colorbar_label: str,
    filename: str = "ksd_heatmap_continuous.pdf",
    grid_res: int = 200,
    log_x: bool = False,
    log_y: bool = False,
    method: str = "cubic",
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
):
    pc = getattr(plot_cfg, "plot", plot_cfg)
    param_names = getattr(pc, "param_latex_names", {"logr": "logr", "logK": "logK"})
    figsize = (plot_cfg.plot.figure.size["width"], plot_cfg.plot.figure.size["height"])
    dpi = plot_cfg.plot.figure["dpi"]
    labelsize = plot_cfg.plot.font["size"]
    tick_labelsize = getattr(pc, "tick_labelsize", 10)
    fontfamily = getattr(pc, "fontfamily", None)
    cbar_labelsize = getattr(pc, "colorbar_labelsize", labelsize)
    vmin = getattr(pc, "vmin", None)
    vmax = getattr(pc, "vmax", None)
    auto_contrast = getattr(pc, "auto_contrast_percentiles", None)  # e.g., (2, 98)
    cmap = _resolve_cmap_from_cfg(plot_cfg)

    # Flatten input
    xs, ys, zs = [], [], []
    for bf, r_dict in data_dict.items():
        for r, val in r_dict.items():
            xs.append(float(bf))
            ys.append(float(r))
            zs.append(float(val))
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    zs = np.array(zs, dtype=float)
    zs = np.log10(zs)

    if log_x and np.any(xs <= 0):
        raise ValueError("log_x=True but some basis function numbers are <= 0.")
    if log_y and np.any(ys <= 0):
        raise ValueError("log_y=True but some radii are <= 0.")

    # Display grid
    xi_vals = np.logspace(np.log10(xs.min()), np.log10(xs.max()), grid_res) if log_x else np.linspace(xs.min(), xs.max(), grid_res)
    yi_vals = np.logspace(np.log10(ys.min()), np.log10(ys.max()), grid_res) if log_y else np.linspace(ys.min(), ys.max(), grid_res)
    Xi, Yi = np.meshgrid(xi_vals, yi_vals)

    # Interpolation space
    ix  = np.log10(xs) if log_x else xs
    iy  = np.log10(ys) if log_y else ys
    iXi = np.log10(Xi) if log_x else Xi
    iYi = np.log10(Yi) if log_y else Yi

    Zi = griddata((ix, iy), zs, (iXi, iYi), method=method)
    Zmask = np.ma.array(Zi, mask=np.isnan(Zi))

    # Optional automatic contrast to avoid washed-out colors (only if vmin/vmax not set)
    if auto_contrast and (vmin is None or vmax is None):
        lo, hi = auto_contrast
        finite = np.isfinite(zs)
        if finite.any():
            if vmin is None:
                vmin = np.percentile(zs[finite], lo)
            if vmax is None:
                vmax = np.percentile(zs[finite], hi)

    rc = {"font.family": fontfamily} if fontfamily else {}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Smooth image (no contour lines, no grid points)
        im = ax.imshow(
            Zmask,
            extent=[xi_vals.min(), xi_vals.max(), yi_vals.min(), yi_vals.max()],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label, fontsize=cbar_labelsize)
        cbar.ax.tick_params(labelsize=tick_labelsize)

        ax.set_xlabel(xlabel or param_names.get("logK", "logK"), fontsize=labelsize)
        ax.set_ylabel(ylabel or param_names.get("logr", "logr"), fontsize=labelsize)

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        ax.tick_params(axis="both", which="both", labelsize=tick_labelsize)

        if title:
            ax.set_title(title, fontsize=labelsize)

        _save_fig(fig, output_dir, filename, plot_cfg)
        plt.close(fig)

