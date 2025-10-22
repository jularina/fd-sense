import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Mapping, Any
import re
from matplotlib.patches import Rectangle


def likelihood(theta: np.ndarray, x: float, sigma2: float = 1.0) -> np.ndarray:
    return np.exp(-0.5 * (x - theta) ** 2 / sigma2)

def prior_normal(theta: np.ndarray, mu: float = 0.0, var: float = 2.0) -> np.ndarray:
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (theta - mu) ** 2 / var)

def prior_laplace(theta: np.ndarray, mu: float = 0.0, b: float = 1.38) -> np.ndarray:
    return (1.0 / (2 * b)) * np.exp(-np.abs(theta - mu) / b)

def prior_cauchy(theta: np.ndarray, x0: float = 0.0, gamma: float = 4.0) -> np.ndarray:
    return 1.0 / (np.pi * gamma * (1.0 + ((theta - x0) / gamma) ** 2))

def prior_uniform(theta: np.ndarray, a: float = -5.0, b: float = 5.0) -> np.ndarray:
    dens = np.zeros_like(theta)
    mask = (theta >= a) & (theta <= b)
    dens[mask] = 1.0 / (b - a)
    return dens

def normalize_density(theta: np.ndarray, f: np.ndarray) -> np.ndarray:
    area = np.trapz(f, theta)
    return f / area if area > 0 else f

def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    cur = cfg
    for part in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(part, default if part == dotted.split(".")[-1] else None)
        else:
            cur = getattr(cur, part, default if part == dotted.split(".")[-1] else None)
    return cur if cur is not None else default

def _apply_rcparams_from_cfg(plot_cfg: Any) -> None:
    plt.rcParams.update({
        "font.size":  _cfg_get(plot_cfg, "plot.font.size", 12),
        "font.family": _cfg_get(plot_cfg, "plot.font.family", "serif"),
        "text.usetex": _cfg_get(plot_cfg, "plot.font.use_tex", False),
    })

def _new_fig_ax(plot_cfg: Any):
    fig, ax = plt.subplots(
        figsize=(
            float(_cfg_get(plot_cfg, "plot.figure.size.width", 6.0)),
            float(_cfg_get(plot_cfg, "plot.figure.size.height", 3.5)),
        ),
        dpi=int(_cfg_get(plot_cfg, "plot.figure.dpi", 150)),
    )
    # match your style (no top/right spines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # labels from config
    ax.set_xlabel(_cfg_get(plot_cfg, "plot.param_latex_names.mu_0", r"$\theta$"))
    ax.set_ylabel(_cfg_get(plot_cfg, "plot.param_latex_names.prior", "density"))
    if bool(_cfg_get(plot_cfg, "plot.figure.tight_layout", True)):
        plt.tight_layout()
    if bool(_cfg_get(plot_cfg, "plot.y_axis.log_scale", False)):
        ax.set_yscale("log")
    return fig, ax

def _palette(plot_cfg: Any, k: int):
    colors = _cfg_get(plot_cfg, "plot.color_palette.colors", None)
    if not colors or k >= len(colors):
        return None
    return colors[k]


def _label_from_cfg(plot_cfg, key: str, default: str) -> str:
    """
    Try plot.<key>, then plot.param_latex_names.<key>, else default.
    """
    val = _cfg_get(plot_cfg, f"plot.{key}", None)
    if val is None:
        val = _cfg_get(plot_cfg, f"plot.param_latex_names.{key}", None)
    return val if val is not None else default


def plot_reference_shapes_x(
    x_obs: float,
    reference: str,            # "normal" | "uniform" | "cauchy"
    plot_cfg: Any,
    output_dir: str,
    theta_min: float = -30.0,
    theta_max: float = 30.0,
    n_grid: int = 4001,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    _apply_rcparams_from_cfg(plot_cfg)
    fig, ax = _new_fig_ax(plot_cfg)

    baseprior   = _label_from_cfg(plot_cfg, "baseprior", "Reference prior")
    refpost_lbl = _label_from_cfg(plot_cfg, "referenceposterior", "Reference posterior")
    prior_lbl   = _label_from_cfg(plot_cfg, "candidateprior", r"$\pi$")

    theta = np.linspace(theta_min, theta_max, n_grid)
    L = likelihood(theta, x_obs)

    reference = reference.lower().strip()
    if reference == "normal":
        ref_name_tex = r"$=\mathcal{N}(0,2)$"
        ref_prior_dens = prior_normal(theta, 0.0, 2.0)
        cand_specs = [
            (r"$\mathcal{L}(0,0.73)$", lambda th: prior_laplace(th, 0.0, 0.73)),
            (r"$\mathcal{C}(0,0.954)$", lambda th: prior_cauchy(th, 0.0, 0.954)),
        ]
    elif reference == "uniform":
        ref_name_tex = r"$=\mathcal{U}(-5,5)$"
        ref_prior_dens = prior_uniform(theta, -5.0, 5.0)
        cand_specs = [
            (r"$\mathcal{N}(0,2)$",    lambda th: prior_normal(th, 0.0, 2.0)),
            (r"$\mathcal{L}(0,1.38)$", lambda th: prior_laplace(th, 0.0, 0.73)),
            (r"$\mathcal{C}(0,0.954)$", lambda th: prior_cauchy(th, 0.0, 0.954)),
        ]
    elif reference == "cauchy":
        ref_name_tex = r"$=\mathcal{C}(0,4)$"
        ref_prior_dens = prior_cauchy(theta, 0.0, 4.0)
        cand_specs = [
            (r"$\mathcal{N}(0,2)$",    lambda th: prior_normal(th, 0.0, 2.0)),
            (r"$\mathcal{L}(0,1.38)$", lambda th: prior_laplace(th, 0.0, 0.73)),
        ]
    else:
        raise ValueError("reference must be one of: 'normal', 'uniform', 'cauchy'")

    # Reference posterior from unscaled prior and unscaled likelihood
    ref_post_dens = normalize_density(theta, L * ref_prior_dens)

    c0 = _palette(plot_cfg, 0)
    c1 = _palette(plot_cfg, 1)
    extra_colors = [_palette(plot_cfg, 2), _palette(plot_cfg, 3), _palette(plot_cfg, 4), _palette(plot_cfg, 5)]

    # Likelihood (shape only)
    ax.axvline(x_obs, color=c0, linestyle="--",
               label=f"$x={x_obs}$")

    # Reference prior (true PDF) and posterior
    ax.plot(theta, ref_prior_dens, label=f"{baseprior} {ref_name_tex}", color=c1, linewidth=1.7)
    ax.plot(theta, ref_post_dens,  label=f"{refpost_lbl}",            color=c1, linestyle=":", linewidth=1.7)

    # Candidates (true PDF priors + normalized posteriors)
    for i, (prior_tex, prior_fun) in enumerate(cand_specs):
        color_i = extra_colors[i % len(extra_colors)]
        cand_prior_dens = prior_fun(theta)
        cand_post_dens  = normalize_density(theta, L * cand_prior_dens)

        ax.plot(theta, cand_prior_dens, label=f"{prior_lbl}={prior_tex}", color=color_i, linewidth=1.7)
        # ax.plot(theta, cand_post_dens,  label=f"{post_lbl} ({prior_lbl}={prior_tex})", color=color_i, linestyle=":", linewidth=1.7)

    if bool(_cfg_get(plot_cfg, "plot.legend.show", True)):
        ax.legend(frameon=False, loc="upper left")

    filename = f"global_comparison_posterior_shapes_ref_{reference}_x{int(x_obs)}.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_informal_means_x(
    x_obs: float,
    means_dict: Dict[str, float],  # {"label": mean_value}
    plot_cfg: Any,
    output_dir: str,
    theta_min: float = -10.0,
    theta_max: float = 17.0,
    n_grid: int = 4001,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    _apply_rcparams_from_cfg(plot_cfg)
    fig, ax = _new_fig_ax(plot_cfg)

    theta = np.linspace(theta_min, theta_max, n_grid)
    L = likelihood(theta, x_obs)
    L_scaled = L / L.max() if L.max() > 0 else L

    # Likelihood curve
    c0 = _palette(plot_cfg, 0)
    ax.axvline(x_obs, color=c0, linestyle="--",
               label=f"$x={x_obs}$")

    # Map textual prior names to LaTeX forms in labels
    def _prior_texify(lbl: str) -> str:
        if not isinstance(lbl, str):
            return lbl
        lbl = re.sub(r"\bNormal(?:\s+prior)?\b",  r"$\\mathcal{N}(0,2)$",    lbl, flags=re.IGNORECASE)
        lbl = re.sub(r"\bLaplace(?:\s+prior)?\b", r"$\\mathcal{L}(0,0.73)$", lbl, flags=re.IGNORECASE)
        lbl = re.sub(r"\bCauchy(?:\s+prior)?\b",  r"$\\mathcal{C}(0,0.954)$",    lbl, flags=re.IGNORECASE)
        return lbl

    # Vertical mean markers (use palette entries after index 0)
    idx = 1
    for raw_label, mu in means_dict.items():
        label = _prior_texify(raw_label)
        ax.axvline(
            mu,
            linestyle=":",
            color=_palette(plot_cfg, idx),
            label=label,
        )
        idx += 1

    if bool(_cfg_get(plot_cfg, "plot.legend.show", True)):
        ax.legend(frameon=False)

    filename = f"global_comparison_informal_means_x{int(x_obs)}.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return save_path

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

def plot_prior_range_comparison_split(
    wim_cauchy_scales_beta,
    x0_value=0,
    mu0_laplace=0,
    normal_mu_range=(-5, 5),
    normal_sigma_range=(5, 10),
    normal_mu_points=6,
    normal_sigma_points=3,
    laplace_b_range=(3, 14),
    laplace_b_points=20,
    plot_cfg=None,
    output_dir=".",
    filenames=(
        "prior_range_wim_cauchy.pdf",
        "prior_range_ksd_normal.pdf",
        "prior_range_ksd_laplace.pdf",
    ),
    normal_fill_alpha=0.12,
):
    """Create three separate plots for (a) WIM/Cauchy, (b) KSD/Normal, (c) KSD/Laplace."""
    # Apply your plotting RC and pull sizing/colors
    _apply_plot_rc(plot_cfg)
    W = plot_cfg.plot.figure.size.width
    H = plot_cfg.plot.figure.size.height
    dpi = plot_cfg.plot.figure.dpi
    base_color = plot_cfg.plot.color_palette.colors[0]  # shared color

    # ------------------------------ (a) WIM / Cauchy dots ------------------------------
    fig_a, ax_a = plt.subplots(1, 1, figsize=(W, H), dpi=dpi)
    scales = np.asarray(wim_cauchy_scales_beta, dtype=float)
    ax_a.plot(scales, np.zeros_like(scales), linestyle="none",
              marker=".", markersize=10, color="black")
    pad = 0.1 * (scales.max() - scales.min() if scales.size else 1.0)
    xmin = max(0.0, (scales.min() - pad) if scales.size else 0.0)
    xmax = (scales.max() + pad) if scales.size else 1.0
    ax_a.set_xlim(xmin, xmax)
    ax_a.set_ylim(-0.5, 0.5)
    ax_a.set_ylabel(r"$x_0$")

    ax_a.set_xlabel(r"$\gamma_0$")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    _save_fig(fig_a, output_dir, filenames[0], plot_cfg)

    # ------------------------------ (b) KSD / Normal grid ------------------------------
    fig_b, ax_b = plt.subplots(1, 1, figsize=(W, H), dpi=dpi)
    mu_min, mu_max = normal_mu_range
    s_min, s_max = normal_sigma_range
    mu_vals = np.linspace(mu_min, mu_max, max(2, int(normal_mu_points)))
    sigma_vals = np.linspace(s_min, s_max, max(2, int(normal_sigma_points)))
    M, S = np.meshgrid(mu_vals, sigma_vals)

    # Fill rectangle (area) with config color and alpha
    rect = Rectangle((mu_min, s_min), mu_max - mu_min, s_max - s_min,
                     facecolor=base_color, edgecolor="none", alpha=normal_fill_alpha)
    ax_b.add_patch(rect)
    ax_b.plot([mu_min, mu_max, mu_max, mu_min, mu_min],
              [s_min,  s_min,  s_max,  s_max,  s_min],
              color=base_color, linewidth=1)
    corner_pts = [(mu_min, s_min), (mu_min, s_max), (mu_max, s_min), (mu_max, s_max)]
    for (xm, ys) in corner_pts:
        if xm == mu_min and ys == s_min:
            ax_b.plot(xm, ys, marker="*", color="red", markersize=10, zorder=5)
        else:
            ax_b.plot(xm, ys, marker=".", color="black", markersize=10, zorder=5)

    ax_b.set_xlabel(r"$\mu_0$")
    ax_b.set_ylabel(r"$\sigma_0$")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    _save_fig(fig_b, output_dir, filenames[1], plot_cfg)

    # ------------------------------ (c) KSD / Laplace line + corners -------------------
    fig_c, ax_c = plt.subplots(1, 1, figsize=(W, H), dpi=dpi)
    b_min, b_max = laplace_b_range
    b_vals = np.linspace(b_min, b_max, max(2, int(laplace_b_points)))

    # constant μ0 (argument of the function, e.g. mu0_value=0)
    y_val = mu0_laplace * np.ones_like(b_vals)

    # Colored line segment along the range using config color
    ax_c.plot([b_min, b_max], [mu0_laplace, mu0_laplace],
              linestyle="-", linewidth=2.0, color=base_color)

    # Corner stars in red
    ax_c.plot(b_min, mu0_laplace, marker=".", color="black", markersize=10, zorder=5)
    ax_c.plot(b_max, mu0_laplace, marker="*", color="red", markersize=10, zorder=5)

    # Axes
    ax_c.set_xlim(b_min - 0.1 * (b_max - b_min), b_max + 0.1 * (b_max - b_min))
    ax_c.set_ylim(mu0_laplace - 0.5, mu0_laplace + 0.5)
    ax_c.set_xlabel(r"$b_0$")
    ax_c.set_ylabel(r"$\mu_0$")

    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    _save_fig(fig_c, output_dir, filenames[2], plot_cfg)