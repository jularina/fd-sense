import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Mapping, Any
import re


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

    like_lbl    = _label_from_cfg(plot_cfg, "likelihood", "Likelihood (unit-peak)")
    baseprior   = _label_from_cfg(plot_cfg, "baseprior", "Reference prior")
    refpost_lbl = _label_from_cfg(plot_cfg, "referenceposterior", "Reference posterior")
    post_lbl    = _label_from_cfg(plot_cfg, "candidateposterior", "Posterior")
    prior_lbl   = _label_from_cfg(plot_cfg, "candidateprior", r"$\pi$")

    theta = np.linspace(theta_min, theta_max, n_grid)
    L = likelihood(theta, x_obs)

    reference = reference.lower().strip()
    if reference == "normal":
        ref_name_tex = r"$=\mathcal{N}(0,2)$"
        ref_prior_dens = prior_normal(theta, 0.0, 2.0)
        cand_specs = [
            (r"$\mathcal{L}(0,1.38)$", lambda th: prior_laplace(th, 0.0, 1.38)),
            (r"$\mathcal{C}(0,0.954)$", lambda th: prior_cauchy(th, 0.0, 0.954)),
        ]
    elif reference == "uniform":
        ref_name_tex = r"$=\mathcal{U}(-5,5)$"
        ref_prior_dens = prior_uniform(theta, -5.0, 5.0)
        cand_specs = [
            (r"$\mathcal{N}(0,2)$",    lambda th: prior_normal(th, 0.0, 2.0)),
            (r"$\mathcal{L}(0,1.38)$", lambda th: prior_laplace(th, 0.0, 1.38)),
            (r"$\mathcal{C}(0,0.954)$", lambda th: prior_cauchy(th, 0.0, 0.954)),
        ]
    elif reference == "cauchy":
        ref_name_tex = r"$=\mathcal{C}(0,4)$"
        ref_prior_dens = prior_cauchy(theta, 0.0, 4.0)
        cand_specs = [
            (r"$\mathcal{N}(0,2)$",    lambda th: prior_normal(th, 0.0, 2.0)),
            (r"$\mathcal{L}(0,1.38)$", lambda th: prior_laplace(th, 0.0, 1.38)),
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
               label=f"{like_lbl} at $x={x_obs}$")

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

    like_lbl = _label_from_cfg(plot_cfg, "likelihood", "Likelihood")

    theta = np.linspace(theta_min, theta_max, n_grid)
    L = likelihood(theta, x_obs)
    L_scaled = L / L.max() if L.max() > 0 else L

    # Likelihood curve
    c0 = _palette(plot_cfg, 0)
    ax.plot(theta, L_scaled, label=f"{like_lbl}", color=c0)

    # Map textual prior names to LaTeX forms in labels
    def _prior_texify(lbl: str) -> str:
        if not isinstance(lbl, str):
            return lbl
        lbl = re.sub(r"\bNormal(?:\s+prior)?\b",  r"$\\mathcal{N}(0,2)$",    lbl, flags=re.IGNORECASE)
        lbl = re.sub(r"\bLaplace(?:\s+prior)?\b", r"$\\mathcal{L}(0,1.38)$", lbl, flags=re.IGNORECASE)
        lbl = re.sub(r"\bCauchy(?:\s+prior)?\b",  r"$\\mathcal{C}(0,0.954)$",    lbl, flags=re.IGNORECASE)
        return lbl

    # Vertical mean markers (use palette entries after index 0)
    idx = 1
    for raw_label, mu in means_dict.items():
        label = _prior_texify(raw_label)
        ax.axvline(
            mu,
            linestyle=_cfg_get(plot_cfg, "plot.lines.mean_linestyle", "--"),
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