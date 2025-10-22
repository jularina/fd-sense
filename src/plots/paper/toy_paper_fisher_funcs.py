from collections import defaultdict
from typing import List, Tuple, Dict, FrozenSet, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from matplotlib.colors import LinearSegmentedColormap
import colorsys
from matplotlib.colors import to_rgb, Normalize, ListedColormap, BoundaryNorm
import matplotlib.cm as cmx
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.cm import ScalarMappable
from scipy.special import logsumexp
import os
from typing import Any
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from scipy.stats import sem, t

from src.distributions.gaussian import Gaussian


def plot_multivariate_priors_densities(all_params, all_ksds, output_dir, plot_cfg, true_theta=None, true_cov=None):
    """
    Plots joint prior densities (3 of them) with marginals using fixed 3-color scheme and KSD arrow bar.
    """
    os.makedirs(output_dir, exist_ok=True)


    sorted_keys = sorted(all_ksds, key=lambda k: all_ksds[k])
    palette_colors = plot_cfg.plot.color_palette.colors[:3]
    rgb_colors = [to_rgb(c) for c in palette_colors[::-1]]

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig = plt.figure(figsize=(3.0, 3.0))
    grid = plt.GridSpec(4, 4, hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(grid[1:, :-1])
    ax_x = fig.add_subplot(grid[0, :-1], sharex=ax_main)
    ax_y = fig.add_subplot(grid[1:, -1], sharey=ax_main)

    for i, key in enumerate(sorted_keys):
        mu = all_params[key]["mu"]
        cov = all_params[key]["cov"]
        color = rgb_colors[i]

        samples = np.random.multivariate_normal(mu, cov, size=2000)

        # Joint KDE
        sns.kdeplot(
            x=samples[:, 0],
            y=samples[:, 1],
            ax=ax_main,
            fill=True,
            levels=10,
            alpha=0.9,
            thresh=0.01,
            linewidths=0.7,
            color=color,
        )

        # Marginals
        sns.kdeplot(samples[:, 0], ax=ax_x, color=color, fill=True, alpha=0.6)
        sns.kdeplot(samples[:, 1], ax=ax_y, color=color, fill=True, alpha=0.6, vertical=True)

    # Plot true_theta reference
    if true_theta is not None:
        # Black lines
        ax_main.axvline(true_theta[0], color="k", linestyle="-", lw=1)
        ax_main.axhline(true_theta[1], color="k", linestyle="-", lw=1)
        ax_x.axvline(true_theta[0], color="k", linestyle="-", lw=1)
        ax_y.axhline(true_theta[1], color="k", linestyle="-", lw=1)

        # Overlay black marginal KDE lines
        true_samples = np.random.multivariate_normal(true_theta, true_cov, size=2000)
        sns.kdeplot(true_samples[:, 0], ax=ax_x, color="k", lw=1, fill=False)
        sns.kdeplot(true_samples[:, 1], ax=ax_y, color="k", lw=1, fill=False, vertical=True)

        sns.kdeplot(
            x=true_samples[:, 0],
            y=true_samples[:, 1],
            ax=ax_main,
            fill=False,
            levels=10,
            alpha=0.3,
            thresh=0.01,
            linewidths=0.7,
            color="black",
        )

    # Axis cleanup
    ax_x.axis("off")
    ax_y.axis("off")
    ax_main.set_xlabel("$\\mu_{01}$")
    ax_main.set_ylabel("$\\mu_{02}$")
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # Add custom arrow colorbar for KSD ordering
    cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
    cmap = ListedColormap(rgb_colors)
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N)
    cb = plt.colorbar(cmx.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)
    cb.ax.tick_params(size=0, width=0, labelsize=0, left=False, right=False)
    cb.set_ticks([])

    # Arrow showing increasing KSD
    cb.ax.annotate(
        '',
        xy=(1.4, 0.7),
        xytext=(1.4, 0.3),
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            lw=1,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=8
        ),
    )

    cb.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriorsShort)

    fig.tight_layout(rect=[0, 0, 0.9, 1.0])
    output_path = os.path.join(output_dir, "multivariate_joint_prior_plot.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

def plot_multi_line_plots(
    ksd_results: Dict[Tuple[float, ...], float],
    param_names: List[str],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    latex_param_names = plot_cfg.plot.param_latex_names
    colors = plot_cfg.plot.color_palette.colors

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    param_values = np.array(list(ksd_results.keys()))
    ksd_values = np.array(list(ksd_results.values()))

    if len(param_names) != 2:
        raise ValueError("This function currently supports exactly two parameters.")

    def make_multi_line_plot(
            fixed_idx: int,
            varying_idx: int,
            filename_prefix: str,
    ):
        fixed_param_name = param_names[fixed_idx]
        varying_param_name = param_names[varying_idx]
        fixed_param_latex = latex_param_names.get(fixed_param_name, fixed_param_name)
        varying_param_latex = latex_param_names.get(varying_param_name, varying_param_name)

        fixed_vals = np.unique(param_values[:, fixed_idx])
        base_rgb = to_rgb(colors[0])  # Base color from palette
        base_hsv = colorsys.rgb_to_hsv(*base_rgb)

        # Compute average KSDs for each fixed value to sort and normalize for brightness
        avg_ksd_per_line = []
        for fixed_val in fixed_vals:
            mask = param_values[:, fixed_idx] == fixed_val
            y = ksd_values[mask]
            avg_ksd_per_line.append(np.mean(y))
        avg_ksd_per_line = np.array(avg_ksd_per_line)

        num_lines = len(fixed_vals)
        brightness_vals = np.linspace(0.2, 0.9, num_lines)

        fig, ax = plt.subplots(
            figsize=(
                plot_cfg.plot.figure.size.width,
                plot_cfg.plot.figure.size.height,
            ),
            dpi=plot_cfg.plot.figure.dpi,
        )

        for i, fixed_val in enumerate(fixed_vals):
            mask = param_values[:, fixed_idx] == fixed_val
            x = param_values[mask, varying_idx]
            y = ksd_values[mask]
            sorted_idx = np.argsort(x)
            x = x[sorted_idx]
            y = y[sorted_idx]

            # Adjust brightness of the base color
            shaded_rgb = colorsys.hsv_to_rgb(base_hsv[0], base_hsv[1], brightness_vals[i])

            ax.plot(
                x, y,
                marker='.',
                label=f"{fixed_param_latex} = {fixed_val:.0f}",
                color=shaded_rgb,
            )

            if getattr(plot_cfg.plot, "show_min_point", False) and fixed_param_latex == "$\\sigma_0$" and fixed_val == 3.0:
                min_idx = np.argmin(y)
                min_x = x[min_idx]
                min_y = y[min_idx]
                ax.scatter(
                    min_x, min_y,
                    color="black",
                    zorder=5,
                    marker='x',
                    s=50,
                )

            if getattr(plot_cfg.plot, "show_max_point", False) and fixed_param_latex == "$\\sigma_0$" and fixed_val == 2.0:
                max_idx = np.argmax(y)
                max_x = x[max_idx]
                max_y = y[max_idx]
                ax.scatter(
                    max_x, max_y,
                    color="red",
                    zorder=6,
                    marker='*',
                    s=50,
                )

        ax.set_xlabel(varying_param_latex)
        ksd_latex = latex_param_names.get("estimatedFDposteriorsShort")
        ylabel = f"log {ksd_latex}" if plot_cfg.plot.y_axis.log_scale else ksd_latex
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right')

        if plot_cfg.plot.y_axis.log_scale:
            ax.set_yscale("log")

        if plot_cfg.plot.figure.tight_layout:
            plt.tight_layout()

        save_path = os.path.join(output_dir, f"{filename_prefix}_{varying_param_name}_vs_{fixed_param_name}.pdf")
        fig.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close(fig)
        print(f"Saved combined KSD vs {varying_param_name} plot to: {save_path}")

    make_multi_line_plot(fixed_idx=1, varying_idx=0, filename_prefix="ksd_multiline")


def plot_single_param(
    ksd_results: Dict[float, float],
    param_name: str,
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    latex_param_names = plot_cfg.plot.param_latex_names
    latex_param_name = plot_cfg.plot.param_latex_names.get(param_name, param_name)

    # Sort the keys for consistent plotting
    x_vals = np.array(sorted(ksd_results.keys()))
    y_vals = np.array([ksd_results[x] for x in x_vals])

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
    })

    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    ax.plot(x_vals, y_vals, marker='.', color=plot_cfg.plot.color_palette.colors[0])

    ax.set_xlabel(latex_param_name)
    ksd_latex = latex_param_names.get("estimatedFDposteriorsShort")
    ylabel = f"log {ksd_latex}" if plot_cfg.plot.y_axis.log_scale else ksd_latex
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    if plot_cfg.plot.y_axis.log_scale:
        ax.set_yscale("log")

    if getattr(plot_cfg.plot, "show_min_point", False):
        min_idx = np.argmin(y_vals)
        min_x = x_vals[min_idx]
        min_y = y_vals[min_idx]
        ax.scatter(
            min_x, min_y,
            color="black",
            zorder=5,
            marker='x',
            s=50,
        )

    if getattr(plot_cfg.plot, "show_max_point", False):
        max_idx = np.argmax(y_vals)
        max_x = x_vals[max_idx]
        max_y = y_vals[max_idx]
        ax.scatter(
            max_x, max_y,
            color="red",
            zorder=6,
            marker='*',
            s=50,
        )

    filename = f"ksd_line_{param_name}.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close(fig)

    print(f"Saved line plot to: {save_path}")


def plot_eta_surface(
    results: List[Tuple[Dict[str, float], np.ndarray, float]],
    corner_points: List[Dict[str, float]],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import LinearLocator
    os.makedirs(output_dir, exist_ok=True)
    latex_param_names = plot_cfg.plot.param_latex_names

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
    })

    def _key_from_params(p: Dict[str, float]) -> FrozenSet[Tuple[str, float]]:
        return frozenset({(k, float(f"{v:.8f}")) for k, v in p.items()})

    def _fmt(v: float) -> str:
        return f"{float(v):.3g}"

    def _get_mu_sigma_from_corner(cp: Dict[str, float]) -> Tuple[float, float]:
        def _first(cp, keys):
            for k in keys:
                if k in cp:
                    return cp[k]
            return None
        mu = _first(cp, ["mu_0", "mu0", "mu"])
        sg = _first(cp, ["sigma_0", "sigma0", "sigma", "std", "sd"])
        return mu, sg

    def _ensure_math(s: str) -> str:
        return s if "$" in s else f"${s}$"

    # labels from config
    try:
        mu_label = getattr(latex_param_names, "mu_0")
    except Exception:
        mu_label = r"\mu_0"
    try:
        sigma_label = getattr(latex_param_names, "sigma_0")
    except Exception:
        sigma_label = r"\sigma_0"
    mu_label = _ensure_math(mu_label)
    sigma_label = _ensure_math(sigma_label)

    # font sizes from config
    base_fs = int(plot_cfg.plot.font.size)
    corner_num_fs = max(7, int(base_fs * 1.25))
    legend_title_fs = max(7, int(base_fs * 1.05))
    legend_line_fs = max(7, int(base_fs * 1.00))
    corner_dot_size = max(10, int(base_fs * 1.4))  # <-- smaller black dots
    y_label_fs = max(base_fs + 2, int(base_fs * 1.25))  # <-- larger y-label

    # ---------- gather data ----------
    x, y, z = [], [], []
    coords_by_key: Dict[FrozenSet[Tuple[str, float]], Tuple[float, float, float]] = {}

    corner_points = [cp[0] for cp in corner_points]
    corner_keys: List[FrozenSet[Tuple[str, float]]] = [_key_from_params(cp) for cp in corner_points]
    for prior_params, eta_tilde, ksd_est in results:
        if len(eta_tilde) < 2:
            continue
        eta0, eta1 = float(eta_tilde[0]), float(eta_tilde[1])
        x.append(eta0)
        y.append(eta1)
        z_val = float(np.log10(ksd_est) if plot_cfg.plot.y_axis.log_scale else ksd_est)
        z.append(z_val)
        coords_by_key[_key_from_params(prior_params)] = (eta0, eta1, z_val)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    palette_colors = plot_cfg.plot.color_palette.colors[::-1]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", palette_colors)

    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(x, y, z, cmap=cmap, edgecolor="none", linewidth=0.2, antialiased=True)

    ax.set_xlabel(r"$\gamma_0=\frac{\mu_0}{\sigma_0^2}$")
    ax.set_ylabel(r"$\gamma_1=\frac{-0.5}{\sigma_0^2}$", fontsize=y_label_fs)  # <-- larger y-label

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    # ---------- max marker ----------
    max_idx = int(np.argmax(z))
    max_pt = (x[max_idx], y[max_idx], z[max_idx])
    ax.scatter(*max_pt, color="red", marker="*", s=50, zorder=8)

    # ---------- corners: numbers + straight trajectory + black dots ----------
    corner_coords_ordered: List[Tuple[float, float, float]] = []
    legend_lines: List[str] = []

    for idx, (cp, ck) in enumerate(zip(corner_points, corner_keys), start=1):
        if ck not in coords_by_key:
            continue
        cx, cy, cz = coords_by_key[ck]
        corner_coords_ordered.append((cx, cy, cz))

        # number at the corner (small z-lift)
        z_lift = 0.02 * (ax.get_zlim()[1] - ax.get_zlim()[0])
        ax.text(cx, cy, cz + z_lift, f"{idx}",
                fontsize=corner_num_fs, color="black",
                ha="center", va="bottom", zorder=12, weight="bold")

        mu_val, sg_val = _get_mu_sigma_from_corner(cp)
        parts = []
        if mu_val is not None:
            parts.append(f"{mu_label}={_fmt(mu_val)}")
        if sg_val is not None:
            parts.append(f"{sigma_label}={_fmt(sg_val)}")
        legend_lines.append(f"{idx}: " + ", ".join(parts) if parts else f"{idx}: (corner)")

    # straight trajectory
    if len(corner_coords_ordered) >= 2:
        traj = np.array(corner_coords_ordered, dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color="black", linestyle="-", linewidth=1.1, alpha=0.7, zorder=9)

    # black points at corners (skip the red max)
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    tol_x = 1e-6 * max(1.0, (xlim[1] - xlim[0]))
    tol_y = 1e-6 * max(1.0, (ylim[1] - ylim[0]))
    tol_z = 1e-6 * max(1.0, (zlim[1] - zlim[0]))
    for (cx, cy, cz) in corner_coords_ordered:
        is_max_corner = (abs(cx - max_pt[0]) < tol_x) and (abs(cy - max_pt[1])
                                                           < tol_y) and (abs(cz - max_pt[2]) < tol_z)
        if not is_max_corner:
            ax.scatter(cx, cy, cz, color="black", s=corner_dot_size, zorder=11)

    # ---------- middle-right legend ----------
    N = max(1, len(legend_lines))
    height = min(0.60, 0.05 * N + 0.10)
    bottom = 0.50 - height / 2.0
    legend_ax = fig.add_axes([0.77, bottom, 0.21, height])
    legend_ax.axis("off")
    legend_ax.text(0.0, 1.02, "Corners", fontsize=legend_title_fs, fontweight="bold",
                   ha="left", va="bottom")
    ys = [0.5] if N == 1 else np.linspace(0.85, 0.10, N)
    for yv, line in zip(ys, legend_lines):
        legend_ax.text(0.0, float(yv), line, fontsize=legend_line_fs, ha="left", va="center")

    # ---------- ticks: exactly 3 on x, y, z ----------
    ax.xaxis.set_major_locator(LinearLocator(4))
    ax.yaxis.set_major_locator(LinearLocator(4))
    ax.zaxis.set_major_locator(LinearLocator(4))

    # format tick labels with max 2 decimals (strip trailing zeros)
    def _fmt_two_decimals(x, pos):
        s = f"{x:.2f}".rstrip('0').rstrip('.')
        return s

    from matplotlib.ticker import LinearLocator, FuncFormatter
    _tickfmt = FuncFormatter(_fmt_two_decimals)
    ax.xaxis.set_major_formatter(_tickfmt)
    ax.yaxis.set_major_formatter(_tickfmt)
    ax.zaxis.set_major_formatter(_tickfmt)

    # Bring tick labels closer to the plot
    tick_pad = 2  # pixels; lower = closer
    ax.tick_params(axis='x', which='major', pad=tick_pad)
    ax.tick_params(axis='y', which='major', pad=tick_pad)
    ax.tick_params(axis='z', which='major', pad=tick_pad)

    # OPTIONAL (3D-specific): pull ticks further inward if needed
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo['tick']['outward_factor'] = 0.0  # reduce outward push
        a._axinfo['tick']['inward_factor'] = 0.2  # small inward pull

    # ---------- cosmetics ----------
    ax.view_init(elev=30, azim=50)
    ksd_qf_latex = getattr(latex_param_names, "estimatedKSDposteriorsQuadraticForm")
    zmin, zmax = ax.get_zlim()
    xmid = np.max(ax.get_xlim())
    ymin = np.min(ax.get_ylim())
    ax.text(xmid, ymin - 0.05, zmax - 0.03, ksd_qf_latex,
            rotation=90, fontsize=base_fs, va="bottom", ha="left")

    # save
    filename = "eta_surface_from_corners.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved 3D eta surface with straight path, smaller black corner points, 3 ticks/axis, and legend to: {save_path}")


def plot_sdp_densities_and_logprior(
    basis_function,
    psi_sdp_list: list[np.ndarray],
    radius_labels: list[float],
    ksd_estimates: list[float],
    prior_distribution,
    plot_cfg,
    output_dir: str,
    domain: tuple = (-5, 5),
    resolution: int = 200,
) -> None:
    """
    Combined plot: top = densities, bottom = log prior.
    Single shared legend (same styling as plot_sdp_comparisons_multiple_radii).
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    # --- Grid & features ---
    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    dx = float(x[1, 0] - x[0, 0])
    Phi_x = basis_function.evaluate(x)  # (resolution, D)
    prior_density_true = prior_distribution.pdf(x).flatten()
    log_prior_true = prior_distribution.log_pdf(x).flatten()

    # --- Colors ---
    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]

    # --- Names & labels ---
    names = plot_cfg.plot.param_latex_names
    ksd_label = names.get("estimatedFDposteriorsShort")
    xlabel = names.get("mu_0", "theta")
    ylabel_density = names.get("nonparametric_prior", "Density")
    ylabel_logprior = names.get("log_prior", "log_prior")
    true_density_label = names.get("logbaseprior", "True Prior Density")
    true_logprior_label = names.get("logbaseprior")
    true_prior_label = names.get("baseprior")
    approx_sym = r"$\approx$"
    geq_sym = r"$\geq$"

    # --- Figure & Axes ---
    fig, (ax_density, ax_log) = plt.subplots(
        2, 1,
        figsize=(plot_cfg.plot.figure.size.width,
                 plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
        sharex=True
    )

    # ===== Top panel: densities =====
    # SDP densities
    for i, (psi, r_label, ksd) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
        f = (Phi_x @ psi).flatten()
        logZ = logsumexp(f) + np.log(dx)
        p_hat = np.exp(f - logZ)
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label} ({ksd:.1f})"
        ax_density.plot(
            x.flatten(),
            p_hat,
            label=label,
            linewidth=1.5,
            color=color,
        )

    ax_density.set_ylabel(ylabel_density)
    ax_density.grid(True, alpha=0.3)
    ax_density.spines["top"].set_visible(False)
    ax_density.spines["right"].set_visible(False)

    # True prior density
    ax_density.plot(
        x.flatten(),
        prior_density_true,
        label=true_prior_label,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    # ===== Bottom panel: log prior =====

    # SDP log prior approximations (centered)
    for i, (psi, r_label, _) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
        f = (Phi_x @ psi).flatten()
        c = float(np.mean(log_prior_true - f))  # match mean
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label}"
        ax_log.plot(
            x.flatten(),
            f + c,
            label=label,
            linewidth=1.5,
            color=color,
        )

    ax_log.set_xlabel(xlabel)
    ax_log.set_ylabel(ylabel_logprior)
    ax_log.grid(True, alpha=0.3)
    ax_log.spines["top"].set_visible(False)
    ax_log.spines["right"].set_visible(False)

    # True log prior
    ax_log.plot(
        x.flatten(),
        log_prior_true,
        label=true_prior_label,
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    # ===== Shared legend =====
    handles, labels = ax_density.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        title=ksd_label,
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        frameon=False,
        labelspacing=0.4,
        handlelength=1.8,
        handletextpad=0.5,
        borderpad=0.4,
    )
    for t in leg.get_texts():
        t.set_wrap(True)
    leg.get_title().set_ha("right")
    leg._legend_box.align = "right"
    plt.setp(leg.get_texts(), fontsize=plt.rcParams["font.size"] * 0.9)
    plt.setp(leg.get_title(), fontsize=plt.rcParams["font.size"] * 0.9)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout(rect=[0, 0, 0.95, 1])

    # ===== Save =====
    filename = "toy_gaussian_model_nonparametric_optimisation_densities_logprior.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)