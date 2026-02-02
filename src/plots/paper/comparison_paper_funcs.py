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
    wim_cauchy_scales_alpha,
    eta_1_alpha_range,
    eta_2_alpha_range,
    eta_1_beta_range,
    eta_2_beta_range,
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
    fig_a, (ax_a_alpha, ax_a_beta) = plt.subplots(
        2, 1, figsize=(W, H), dpi=dpi, sharex=True
    )

    # --- Alpha ---
    scales_alpha = np.asarray(wim_cauchy_scales_alpha, dtype=float)
    ax_a_alpha.plot(
        scales_alpha,
        np.zeros_like(scales_alpha),
        linestyle="none",
        marker=".",
        markersize=10,
        color="black",
    )
    ax_a_alpha.set_title(r"$\alpha$", pad=4)
    ax_a_alpha.set_ylabel(r"$\mu_\alpha$")
    ax_a_alpha.spines["top"].set_visible(False)
    ax_a_alpha.spines["right"].set_visible(False)
    ax_a_alpha.plot(10, 0, marker="*", color="red", markersize=10, zorder=5)

    # --- Beta ---
    scales_beta = np.asarray(wim_cauchy_scales_beta, dtype=float)
    ax_a_beta.plot(
        scales_beta,
        np.zeros_like(scales_beta),
        linestyle="none",
        marker=".",
        markersize=10,
        color="black",
    )
    ax_a_beta.plot(2.5, 0, marker="*", color="red", markersize=10, zorder=5)
    ax_a_beta.set_title(r"$\beta$", pad=4)
    ax_a_beta.set_ylabel(r"$\mu_\beta$")
    ax_a_beta.set_xlabel(r"$\gamma_\beta$")
    ax_a_beta.spines["top"].set_visible(False)
    ax_a_beta.spines["right"].set_visible(False)

    # --- Shared limits ---
    all_scales = np.concatenate(
        [scales_alpha, scales_beta]
    ) if (scales_alpha.size and scales_beta.size) else (
        scales_alpha if scales_alpha.size else scales_beta
    )

    pad = 0.1 * (all_scales.max() - all_scales.min() if all_scales.size else 1.0)
    xmin = max(0.0, (all_scales.min() - pad) if all_scales.size else 0.0)
    xmax = (all_scales.max() + pad) if all_scales.size else 1.0

    for ax in (ax_a_alpha, ax_a_beta):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.5, 0.5)

    _save_fig(fig_a, output_dir, filenames[0], plot_cfg)

    # ------------------------------ (b) FD / Normal grid ------------------------------
    fig_b, (ax_b_alpha, ax_b_beta) = plt.subplots(
        2, 1, figsize=(W, H), dpi=dpi, sharex=False
    )

    def _draw_eta_box(ax, eta1_range, eta2_range, star_xy, title):
        e1_min, e1_max = eta1_range
        e2_min, e2_max = eta2_range

        # filled rectangle
        rect = Rectangle(
            (e1_min, e2_min),
            e1_max - e1_min,
            e2_max - e2_min,
            facecolor=base_color,
            edgecolor="none",
            alpha=normal_fill_alpha,
        )
        ax.add_patch(rect)

        # outline
        ax.plot(
            [e1_min, e1_max, e1_max, e1_min, e1_min],
            [e2_min, e2_min, e2_max, e2_max, e2_min],
            color=base_color,
            linewidth=1,
        )

        # corners: star at specified corner, dots at the others
        corners = [(e1_min, e2_min), (e1_min, e2_max), (e1_max, e2_min), (e1_max, e2_max)]
        sx, sy = star_xy
        for (x, y) in corners:
            if np.isclose(x, sx) and np.isclose(y, sy):
                ax.plot(x, y, marker="*", color="red", markersize=10, zorder=5)
            else:
                ax.plot(x, y, marker=".", color="black", markersize=10, zorder=5)

        ax.set_title(title, pad=4)
        ax.set_ylabel(r"$\eta_2$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Alpha: star at (-0.12, -0.02)
    _draw_eta_box(
        ax=ax_b_alpha,
        eta1_range=eta_1_alpha_range,
        eta2_range=eta_2_alpha_range,
        star_xy=(-0.12, -0.02),
        title=r"$\alpha$",
    )

    # Beta: star at (-5.12, -1.28)
    _draw_eta_box(
        ax=ax_b_beta,
        eta1_range=eta_1_beta_range,
        eta2_range=eta_2_beta_range,
        star_xy=(-5.12, -1.28),
        title=r"$\beta$",
    )

    ax_b_beta.set_xlabel(r"$\eta_1$")
    _save_fig(fig_b, output_dir, filenames[1], plot_cfg)