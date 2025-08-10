import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from omegaconf import DictConfig
from matplotlib.colors import LinearSegmentedColormap
import colorsys
from matplotlib.colors import to_rgb, Normalize, ListedColormap, BoundaryNorm
import matplotlib.cm as cmx
from matplotlib.patches import Ellipse
from typing import Set, FrozenSet
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable

from src.distributions.gaussian import Gaussian


def plot_ksd_heatmaps(
    ksd_results: Dict[Tuple[float, ...], float],
    param_names: List[str],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    """
    Plots and saves KSD heatmaps using styling from a plot config.

    Args:
        ksd_results: Dictionary mapping parameter value tuples to KSD values.
        param_names: List of parameter names corresponding to the tuple order.
        plot_cfg: Plot configuration (e.g., loaded from YAML and converted to DictConfig).
        output_dir: Directory to save plots.
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Apply global matplotlib and seaborn settings
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
    })
    sns.set_style(plot_cfg.plot.seaborn.style)
    sns.set_context(plot_cfg.plot.seaborn.context)
    latex_param_names = plot_cfg.plot.param_latex_names

    # Create custom color palette
    colors = plot_cfg.plot.color_palette.colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Extract values
    num_params = len(param_names)
    param_values = np.array(list(ksd_results.keys()))
    ksd_values = np.array(list(ksd_results.values()))

    # Loop through parameter pairs
    for i in range(num_params):
        for j in range(i + 1, num_params):
            x_vals = param_values[:, i]
            y_vals = param_values[:, j]

            xi = np.unique(x_vals)
            yi = np.unique(y_vals)
            zi = np.full((len(yi), len(xi)), np.nan)

            # Fill heatmap grid
            for x, y, ksd in zip(x_vals, y_vals, ksd_values):
                x_idx = np.where(xi == x)[0][0]
                y_idx = np.where(yi == y)[0][0]
                zi[y_idx, x_idx] = ksd

            # Plot
            x_label = latex_param_names.get(param_names[i], param_names[i])
            y_label = latex_param_names.get(param_names[j], param_names[j])
            fig, ax = plt.subplots(
                figsize=(
                    plot_cfg.plot.figure.size.width,
                    plot_cfg.plot.figure.size.height,
                ),
                dpi=plot_cfg.plot.figure.dpi,
            )

            sns.heatmap(
                zi,
                xticklabels=np.round(xi, 3),
                yticklabels=np.round(yi, 3),
                cmap=cmap,
                annot=True,
                fmt=".3f",
                ax=ax,
            )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"KSD heatmap: {x_label} vs {y_label}")

            if plot_cfg.plot.figure.tight_layout:
                plt.tight_layout()

            filename = f"ksd_heatmap_{param_names[i]}_vs_{param_names[j]}.pdf"
            save_path = os.path.join(output_dir, filename)
            fig.savefig(save_path, format="pdf", bbox_inches='tight')
            plt.close(fig)

            print(f"Saved heatmap to: {save_path}")


def plot_ksd_eta_surface(
    results: List[Tuple[Dict[str, float], np.ndarray, float]],
    corner_points: List[Dict[str, float]],  # <-- NEW ARG
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    latex_param_names = plot_cfg.plot.param_latex_names

    x, y, z = [], [], []
    annotations = []

    # Convert corner points to a set of frozensets for fast lookup
    corner_set: Set[FrozenSet[Tuple[str, float]]] = {
        frozenset({(k, float(f"{v:.8f}")) for k, v in cp.items()})
        for cp in corner_points
    }

    for prior_params, eta_tilde, ksd_est in results:
        if len(eta_tilde) < 2:
            continue  # Skip if not enough dimensions

        eta_0, eta_1 = eta_tilde[0], eta_tilde[1]
        x.append(eta_0)
        y.append(eta_1)
        z_val = np.log10(ksd_est) if plot_cfg.plot.y_axis.log_scale else ksd_est
        z.append(z_val)

        # Format prior_params to match precision and check against corners
        param_key = frozenset({(k, float(f"{v:.8f}")) for k, v in prior_params.items()})
        if param_key in corner_set:
            label = "\n".join(
                f"{latex_param_names.get(k+"_0", k)}={v}" for k, v in prior_params.items()
            )
        else:
            label = None
        annotations.append(label)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Create colormap from config
    palette_colors = plot_cfg.plot.color_palette.colors
    palette_colors = palette_colors[::-1]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", palette_colors)

    # Plot
    fig = plt.figure(
        figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height),
        dpi=plot_cfg.plot.figure.dpi,
    )
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(x, y, z, cmap=cmap, edgecolor='none', linewidth=0.2, antialiased=True)

    ax.set_xlabel(r"$\gamma_0=\frac{\mu_0}{\sigma_0^2}$")
    ax.set_ylabel(r"$\gamma_1=\frac{-0.5}{\sigma_0^2}$")

    # ax.set_zlabel(ksd_qf_latex)

    # Annotate only corner points
    for i in range(len(x)):
        if annotations[i] is not None:
            ax.text(x[i], y[i], z[i], annotations[i], fontsize=3, rotation=30, color='black', zorder=10)

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    # Mark min and max KSD points
    z = np.array(z)
    min_idx = np.argmin(z)
    max_idx = np.argmax(z)

    min_x, min_y = x[min_idx], y[min_idx]
    max_x, max_y = x[max_idx], y[max_idx]

    # Red star at maximum
    ax.scatter(
        max_x, max_y, z[max_idx],
        color="red",
        zorder=6,
        marker='*',
        s=50,
    )
    ax.view_init(elev=30, azim=50)

    ksd_qf_latex = latex_param_names.get("estimatedKSDposteriorsQuadraticForm", "KSD")
    zmin, zmax = ax.get_zlim()
    xmid = np.max(ax.get_xlim())
    ymid = np.min(ax.get_ylim())
    ax.text(
        xmid, ymid-0.05, zmax-0.03,
        ksd_qf_latex,
        rotation=90,
        fontsize=10,
        va='bottom',
        ha='left'
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)

    # cbar = fig.colorbar(surf, ax=ax)
    # cbar.ax.tick_params(
    #     size=0,
    #     width=0,
    #     length=0,
    #     labelsize=0,
    #     left=False,
    #     right=False
    # )
    # cbar.set_ticks([])  # Also ensure no major ticks
    # cbar.set_label(ksd_latex)
    #
    # # Add subtle upward arrow inside the colorbar
    # cbar.ax.annotate(
    #     '',
    #     xy=(1.4, 0.6),
    #     xytext=(1.4, 0.4),
    #     xycoords='axes fraction',
    #     textcoords='axes fraction',
    #     arrowprops=dict(
    #         arrowstyle='->',
    #         color='black',
    #         lw=1,
    #         shrinkA=0,
    #         shrinkB=0,
    #         mutation_scale=8  # smaller arrow head
    #     ),
    # )

    filename = "ksd_eta_surface_from_corners.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close(fig)

    print(f"Saved 3D eta surface with annotations to: {save_path}")


def plot_ksd_line_plots(
    ksd_results: Dict[Tuple[float, ...], float],
    param_names: List[str],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    latex_param_names = plot_cfg.plot.param_latex_names

    # Extract all unique values for each parameter
    param_values = np.array(list(ksd_results.keys()))
    ksd_values = np.array(list(ksd_results.values()))
    num_params = len(param_names)

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
    })

    for fixed_idx in range(num_params):
        # The other param index(s)
        varying_idx = [i for i in range(num_params) if i != fixed_idx]

        fixed_param_name = param_names[fixed_idx]
        fixed_param_latex = latex_param_names.get(fixed_param_name, fixed_param_name)

        for fixed_val in np.unique(param_values[:, fixed_idx]):
            # Filter keys where fixed param == fixed_val
            mask = param_values[:, fixed_idx] == fixed_val

            # For each varying param (usually just one for 2D case)
            for v_idx in varying_idx:
                varying_param_name = param_names[v_idx]
                varying_param_latex = latex_param_names.get(varying_param_name, varying_param_name)

                x_vals = param_values[mask, v_idx]
                y_vals = ksd_values[mask]

                # Sort by x for nicer lines
                sorted_idx = np.argsort(x_vals)
                x_vals = x_vals[sorted_idx]
                y_vals = y_vals[sorted_idx]

                fig, ax = plt.subplots(
                    figsize=(
                        plot_cfg.plot.figure.size.width,
                        plot_cfg.plot.figure.size.height,
                    ),
                    dpi=plot_cfg.plot.figure.dpi,
                )

                ax.plot(x_vals, y_vals, marker='.', color=plot_cfg.plot.color_palette.colors[0])

                ax.set_xlabel(varying_param_latex)
                ylabel = "log KSD" if plot_cfg.plot.y_axis.log_scale else "KSD"
                ax.set_ylabel(ylabel)
                ax.set_title(f"KSD vs {varying_param_latex} ({fixed_param_latex} = {fixed_val:.1f})")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                if plot_cfg.plot.figure.tight_layout:
                    plt.tight_layout()

                if plot_cfg.plot.y_axis.log_scale:
                    ax.set_yscale("log")

                filename = f"ksd_line_{fixed_param_name}_{fixed_val:.2f}_vs_{varying_param_name}.pdf"
                save_path = os.path.join(output_dir, filename)
                fig.savefig(save_path, format="pdf", bbox_inches='tight')
                plt.close(fig)

                print(f"Saved line plot to: {save_path}")


def plot_ksd_single_param(
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
    ksd_latex = latex_param_names.get("estimatedKSDposteriors", "KSD")
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


def plot_ksd_multi_line_plots(
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
        "text.latex.preamble": r"""
            \usepackage{amsmath}
        """,
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

        # Normalize inverse brightness: low KSD = high brightness
        # norm = (avg_ksd_per_line - np.min(avg_ksd_per_line)) / (
        #             np.max(avg_ksd_per_line) - np.min(avg_ksd_per_line) + 1e-12)
        # brightness_vals = 0.3 + 0.7 * (1-norm)  # range [0.3, 1.0]
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

            if getattr(plot_cfg.plot, "show_min_point", False) and fixed_param_latex == "$\\sigma_0$" and fixed_val == 2.0:
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
                    marker='*',  # or '^', 's', etc.
                    s=50,
                )

        ax.set_xlabel(varying_param_latex)
        ksd_latex = latex_param_names.get("estimatedKSDposteriors", "KSD")
        ylabel = f"log {ksd_latex}" if plot_cfg.plot.y_axis.log_scale else ksd_latex
        ax.set_ylabel(ylabel)
        # title = f"{ksd_latex} vs {varying_param_latex}"
        # ax.set_title(title)

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

    # Plot 1: Vary param1, lines for each fixed param2
    make_multi_line_plot(fixed_idx=1, varying_idx=0, filename_prefix="ksd_multiline")

    # Plot 2: Vary param2, lines for each fixed param1
    make_multi_line_plot(fixed_idx=0, varying_idx=1, filename_prefix="ksd_multiline")


def plot_gaussian_prior_densities_by_ksd(
    ksd_results: Dict[Tuple[float, ...], float],
    param_names: List[str],
    cfg: DictConfig,
    plot_cfg: DictConfig,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # Base prior
    base_mu = cfg.data.base_prior.mu
    base_sigma = cfg.data.base_prior.sigma
    base_dist = Gaussian(mu=base_mu, sigma=base_sigma)

    # X-axis for PDF evaluation
    x = np.linspace(base_mu - 6 * base_sigma, base_mu + 6 * base_sigma, 300)

    # Apply plot config
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"""
            \usepackage{amsmath}
        """,
    })

    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Plot base prior
    ax.plot(
        x,
        base_dist.pdf(x),
        label="Base prior",
        color="black",
        linewidth=1,
        linestyle="--"
    )

    # KSD values and quantile bins
    ksd_vals = np.array(list(ksd_results.values()))
    sorted_keys = list(ksd_results.keys())
    quantiles = np.quantile(ksd_vals, [1/3, 2/3])

    # Use first 3 colors from color palette and reverse for KSD mapping (higher KSD → darker)
    palette_colors = plot_cfg.plot.color_palette.colors[:3]
    rgb_colors = [to_rgb(c) for c in palette_colors][::-1]  # reverse order

    # Assign color bins based on KSD
    color_bins = []
    for ksd in ksd_vals:
        if ksd <= quantiles[0]:
            color_bins.append(0)
        elif ksd <= quantiles[1]:
            color_bins.append(1)
        else:
            color_bins.append(2)

    # Plot densities with filled color
    for (key, bin_idx) in zip(sorted_keys, color_bins):
        mu_val, sigma_val = key
        dist = Gaussian(mu=mu_val, sigma=sigma_val)
        ax.fill_between(
            x,
            dist.pdf(x),
            color=rgb_colors[bin_idx],
            alpha=0.7,
            linewidth=0
        )

    # Axes labels and formatting
    ax.set_xlabel(plot_cfg.plot.param_latex_names.mu_0)
    ax.set_ylabel(plot_cfg.plot.param_latex_names.prior)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="upper right")

    # Create colorbar without ticks
    cmap = ListedColormap(rgb_colors)
    bounds = [0, 1, 2, 3]
    norm = BoundaryNorm(bounds, cmap.N)
    sm = cmx.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar with no ticks or tick lines
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.tick_params(
        size=0,
        width=0,
        length=0,
        labelsize=0,
        left=False,
        right=False
    )
    cbar.set_ticks([])  # Also ensure no major ticks
    cbar.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriors)

    # Add subtle upward arrow inside the colorbar
    cbar.ax.annotate(
        '',
        xy=(1.4, 0.6),
        xytext=(1.4, 0.4),
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            lw=1,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=8  # smaller arrow head
        ),
    )

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    output_path = os.path.join(output_dir, "prior_densities_by_ksd.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prior density plot with discrete color bar to: {output_path}")


def plot_prior_densities_by_ksd(
    all_ksd_data: Dict[str, Dict],
    cfg: DictConfig,
    plot_cfg: DictConfig,
    output_dir: str,
):
    """
    Plots all prior densities (from different distributions) in one figure,
    using 4 discrete color bins based on KSD quantiles and a colorbar legend.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Setup figure
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Plot base prior
    base_mu = cfg.data.base_prior.mu
    base_sigma = cfg.data.base_prior.sigma
    from src.distributions.gaussian import Gaussian
    base_dist = Gaussian(mu=base_mu, sigma=base_sigma)
    x = np.linspace(base_mu - 6 * base_sigma, base_mu + 6 * base_sigma, 300)
    ax.plot(
        x,
        base_dist.pdf(x),
        label="Base prior",
        color="black",
        linewidth=1,
        linestyle="--"
    )

    # Collect all KSD values
    all_ksd_values = []
    for dist_data in all_ksd_data.values():
        all_ksd_values.extend(dist_data["ksd"].values())
    all_ksd_values = np.array(all_ksd_values)

    # Compute quartile cutoffs
    quantiles = np.quantile(all_ksd_values, [0.25, 0.5, 0.75])

    # Use first 4 colors from palette (reversed if desired)
    palette_colors = plot_cfg.plot.color_palette.colors[:4]
    rgb_colors = [to_rgb(c) for c in palette_colors[::-1]]

    # Plot densities with 4-bin color coding
    for dist_name, dist_data in all_ksd_data.items():
        ksd_results = dist_data["ksd"]
        param_names = dist_data["param_names"]
        distribution_cls = dist_data["distribution_cls"]

        for param_tuple, ksd in ksd_results.items():
            # Determine bin
            if ksd <= quantiles[0]:
                bin_idx = 0
            elif ksd <= quantiles[1]:
                bin_idx = 1
            elif ksd <= quantiles[2]:
                bin_idx = 2
            else:
                bin_idx = 3

            param_dict = dict(zip([p.replace("_0", "") for p in param_names], param_tuple))

            try:
                dist = distribution_cls(**param_dict)
                pdf_vals = dist.pdf(x)
                ax.fill_between(
                    x,
                    pdf_vals,
                    color=rgb_colors[bin_idx],
                    alpha=0.8,
                    linewidth=0.7
                )
            except Exception as e:
                print(f"[WARN] Skipping {dist_name} with params {param_dict}: {e}")

    # Axes formatting
    ax.set_xlabel(plot_cfg.plot.param_latex_names.mu_0)
    ax.set_ylabel(plot_cfg.plot.param_latex_names.prior)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create colorbar
    cmap = ListedColormap(rgb_colors)
    bounds = [0, 1, 2, 3, 4]
    norm = BoundaryNorm(bounds, cmap.N)
    sm = cmx.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.tick_params(size=0, width=0, labelsize=0, left=False, right=False)
    cbar.set_ticks([])
    cbar.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriors)

    # Arrow on colorbar
    cbar.ax.annotate(
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

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    output_path = os.path.join(output_dir, "prior_densities_by_ksd_ALL.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved combined prior density plot: {output_path}")


def plot_ksd_multi_line_plots_with_error_bands(
    ksd_results: Dict[Tuple[float, ...], List[float]],  # Updated to store lists of KSDs
    param_names: List[str],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    latex_param_names = plot_cfg.plot.param_latex_names
    colors = plot_cfg.plot.color_palette.colors

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "figure.dpi": plot_cfg.plot.figure.dpi,
    })

    param_values = np.array(list(ksd_results.keys()))
    ksd_values = np.array(list(ksd_results.values()))

    if len(param_names) != 2:
        raise ValueError("This function currently supports exactly two parameters.")

    fixed_idx, varying_idx = 0, 1  # Fixed = observations_num, Varying = mu_0
    fixed_param_name = param_names[fixed_idx]
    varying_param_name = param_names[varying_idx]
    fixed_param_latex = latex_param_names.get(fixed_param_name, fixed_param_name)
    varying_param_latex = latex_param_names.get(varying_param_name, varying_param_name)

    fixed_vals = np.unique(param_values[:, fixed_idx])
    num_colors = len(colors)

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
        y_lists = ksd_values[mask]

        # Compute the median and ±3 std deviation for each varying parameter value
        medians = [np.median(y_list) for y_list in y_lists]
        lower_bounds = [np.median(y_list) - 3 * np.std(y_list) for y_list in y_lists]
        upper_bounds = [np.median(y_list) + 3 * np.std(y_list) for y_list in y_lists]

        sorted_idx = np.argsort(x)
        x = x[sorted_idx]
        medians = np.array(medians)[sorted_idx]
        lower_bounds = np.array(lower_bounds)[sorted_idx]
        upper_bounds = np.array(upper_bounds)[sorted_idx]

        color = colors[i % num_colors]
        ax.plot(
            x, medians,
            marker='.',
            label=f"{fixed_param_latex} = {int(fixed_val)}",
            color=color,
        )

        # Fill the shaded region for ±3 std deviations
        ax.fill_between(x, lower_bounds, upper_bounds, color=color, alpha=0.3)

        # Highlight minimum of the median curve
        min_idx = np.argmin(medians)
        ax.plot(
            x[min_idx],
            medians[min_idx],
            marker='*',
            markersize=8,
            color=color,
            label=None,
        )

    ylabel = latex_param_names.get("estimatedKSDposteriors", "KSD")
    if plot_cfg.plot.y_axis.log_scale:
        ax.set_yscale("log")
        ylabel = f"log {ylabel}"

    ax.set_xlabel(varying_param_latex)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    save_path = os.path.join(output_dir, f"ksd_multiline_error_bands_{varying_param_name}_vs_{fixed_param_name}.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved KSD multiline plot with error bands to: {save_path}")


def plot_distribution_of_optimal_mu0(
    ksd_results: Dict[Tuple[float, float], List[float]],
    param_names: List[str],
    plot_cfg: DictConfig,
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os

    os.makedirs(output_dir, exist_ok=True)

    latex_param_names = plot_cfg.plot.param_latex_names
    colors = list(plot_cfg.plot.color_palette.colors)

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "figure.dpi": plot_cfg.plot.figure.dpi,
    })

    # Group KSDs by obs_num
    obs_to_mu0_ksd = {}
    for (obs_num, mu_0), ksd_list in ksd_results.items():
        obs_to_mu0_ksd.setdefault(obs_num, {})[mu_0] = ksd_list

    # Flatten all optimal mu_0s into a dataframe
    data = []
    for obs_num, mu0_ksds in obs_to_mu0_ksd.items():
        num_repeats = len(next(iter(mu0_ksds.values())))
        for rep in range(num_repeats):
            min_ksd = float("inf")
            best_mu0 = None
            for mu_0, ksd_list in mu0_ksds.items():
                ksd = ksd_list[rep]
                if ksd < min_ksd:
                    min_ksd = ksd
                    best_mu0 = mu_0
            data.append({'mu_0': best_mu0, 'obs. num.': obs_num})

    df = pd.DataFrame(data)

    # Plot using hue to group by obs_num
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df,
        x='mu_0',
        hue='obs. num.',
        bins=20,
        kde=False,
        multiple="dodge",
        palette=colors,
        ax=ax,
    )

    ax.set_xlabel(latex_param_names.get("mu_0", "mu_0"))
    ax.set_ylabel("Count")
    ax.grid(True)
    ax.legend(title=latex_param_names.get("obs. num.", "obs. num."))

    if plot_cfg.plot.figure.tight_layout:
        plt.tight_layout()

    save_path = os.path.join(output_dir, "optimal_mu0_distribution_across_repeats.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined distribution plot of optimal mu_0s to: {save_path}")


def plot_multivariate_priors_densities_by_ksd(all_params, all_ksds, output_dir, plot_cfg, true_theta=None, true_cov=None):
    """
    Plots joint prior densities (3 of them) with marginals using fixed 3-color scheme and KSD arrow bar.
    """
    assert len(all_params) == 3, "Function currently assumes 3 density entries for coloring."

    os.makedirs(output_dir, exist_ok=True)

    # Sort keys by KSD value ascending
    sorted_keys = sorted(all_ksds, key=lambda k: all_ksds[k])
    sorted_ksds = [all_ksds[k] for k in sorted_keys]

    # Get 3 distinct fixed colors
    palette_colors = plot_cfg.plot.color_palette.colors[:3]
    rgb_colors = [to_rgb(c) for c in palette_colors[::-1]]

    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig = plt.figure(figsize=(6, 6))
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

    cb.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriors)

    fig.tight_layout(rect=[0, 0, 0.9, 1.0])
    output_path = os.path.join(output_dir, "multivariate_joint_prior_plot.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved multivariate joint prior plot to: {output_path}")


def plot_multivariate_joint_prior_densities_by_ksd(results, output_dir, plot_cfg, true_theta=None, true_cov=None):
    """
    Plots joint KDE contours of multivariate priors, colored by KSD magnitude (no fill).
    Overlays true density if provided. Uses color map based on config.
    Highlights the distribution with the largest KSD in red.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort results by KSD ascending (low to high)
    sorted_results = sorted(results, key=lambda x: x[2])
    ksds = [ksd for (_, _, ksd) in sorted_results]
    min_ksd, max_ksd = min(ksds), max(ksds)

    # Normalize KSDs
    norm = Normalize(vmin=min_ksd, vmax=max_ksd)

    # Custom colormap from config
    color_list = plot_cfg.plot.color_palette.colors
    cmap = LinearSegmentedColormap.from_list("ksd_cmap", color_list[::-1])

    # Prepare figure
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    fig, ax = plt.subplots(figsize=(plot_cfg.plot.figure.size.width, plot_cfg.plot.figure.size.height))

    N = 17  # Show top-N and bottom-N KSD priors only
    subset_results = sorted_results[:N] + sorted_results[-N:]

    # Identify the distribution with the maximum KSD
    max_ksd_entry = sorted_results[-1]
    max_ksd_mu = max_ksd_entry[0]["mu"]
    max_ksd_cov = max_ksd_entry[0]["cov"]

    # Plot all selected priors
    for param_dict, _, ksd_est in subset_results:
        mu = param_dict["mu"]
        cov = param_dict["cov"]

        try:
            samples = np.random.multivariate_normal(mu, cov, size=1000)
        except np.linalg.LinAlgError:
            print(f"[WARN] Skipping invalid covariance matrix: {cov}")
            continue

        # If this is the max-KSD distribution, highlight it in red
        is_max_ksd = np.allclose(mu, max_ksd_mu) and np.allclose(cov, max_ksd_cov)

        if is_max_ksd:
            color = "red"
            lw = 2.0
            alpha = 1.0
            levels = 3
            ax.plot(mu[0], mu[1], marker='*', markersize=5, color=color)
        else:
            color = cmap(norm(ksd_est))
            alpha = 0.4 + 0.6 * norm(ksd_est)
            lw = 0.5 + 1.5 * norm(ksd_est)
            levels = 3

        sns.kdeplot(
            x=samples[:, 0],
            y=samples[:, 1],
            ax=ax,
            fill=False,
            levels=levels,
            linewidths=lw,
            color=color,
            alpha=alpha,
        )

    # Overlay true density if provided
    if true_theta is not None and true_cov is not None:
        try:
            true_samples = np.random.multivariate_normal(true_theta, true_cov, size=2000)
            sns.kdeplot(
                x=true_samples[:, 0],
                y=true_samples[:, 1],
                ax=ax,
                fill=False,
                levels=3,
                linewidths=1.2,
                alpha=1.0,
                color="black",
            )
            ax.plot(true_theta[0], true_theta[1], "ko", markersize=5)
            ax.axvline(true_theta[0], color="k", linestyle="--", lw=1)
            ax.axhline(true_theta[1], color="k", linestyle="--", lw=1)
        except np.linalg.LinAlgError:
            print("[WARN] Skipping true density overlay due to invalid covariance.")

    # Labels and appearance
    ax.set_xlabel("$\\mu_{01}$")
    ax.set_ylabel("$\\mu_{02}$")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriors)

    # Save
    output_path = os.path.join(output_dir, "joint_prior_ksd_contours.pdf")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved joint prior KDE (contour) plot to: {output_path}")


def plot_inverse_wishart_scale_ellipses_by_ksd_one_subplot(results, output_dir, plot_cfg):
    """
    Plots 2D ellipses representing inverse Wishart scale matrices,
    colored by KSD value. Highlights the max-KSD distribution in red.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort by KSD (ascending)
    sorted_results = sorted(results, key=lambda x: x[2])
    ksds = [ksd for (_, _, ksd) in sorted_results]
    min_ksd, max_ksd = min(ksds), max(ksds)

    # Normalize KSD values for colormap
    norm = Normalize(vmin=min_ksd, vmax=max_ksd)
    color_list = plot_cfg.plot.color_palette.colors
    cmap = LinearSegmentedColormap.from_list("ksd_cmap", color_list[::-1])

    # Prepare figure and style
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
    fig, ax = plt.subplots(figsize=(plot_cfg.plot.figure.size.width,
                                    plot_cfg.plot.figure.size.height))

    # Parameters for sampling
    step_middle = 5  # Pick every 4th from the middle

    # Slice sorted results
    subset_results = sorted_results[::step_middle] + [sorted_results[-1]]
    n_ellipses = len(subset_results)

    # Identify max-KSD distribution
    max_ksd_entry = sorted_results[-1]
    max_ksd_scale = max_ksd_entry[0]["scale"]

    # Helper to plot ellipses
    def plot_cov_ellipse(cov, pos, ax, nstd=1.5, **kwargs):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellip)

    # Layout for ellipses
    x_spacing = 3.0
    for idx, (param_dict, _, ksd) in enumerate(subset_results):
        scale = param_dict["scale"]
        pos = (idx * x_spacing, 0.0)  # Position ellipses along x-axis

        is_max_ksd = np.allclose(scale, max_ksd_scale)
        if is_max_ksd:
            color = "red"
            lw = 2.5
            alpha = 1.0
        else:
            color = cmap(norm(ksd))
            lw = 1.0
            alpha = 0.7

        try:
            plot_cov_ellipse(scale, pos, ax, edgecolor=color, lw=lw, alpha=alpha, facecolor='none')
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=3)
        except np.linalg.LinAlgError:
            print(f"[WARN] Skipping non-PD matrix: {scale}")

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim(-1, (n_ellipses - 1) * x_spacing + x_spacing)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("")
    ax.set_xticks([])

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Shrink main plot height to make space for colorbar
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + 0.12, box.width, box.height * 0.85])

    # Add horizontal colorbar below the plot
    cbar_height = 0.012
    spacing = 0.09  # vertical spacing between colorbars
    start_y = box.y0 - 0.047  # starting y-position for the first colorbar

    tick_locs = np.linspace(min_ksd+0.01, max_ksd-0.01, 3)
    tick_label_dict = {
        0: ["0.61", "0.63", "0.65"],
        1: ["1.25", "1.4", "1.5"],
        2: ["2.9", "3.2", "3.5"]
    }
    text_labels = {
        0: "$\\nu_0=5$",
        1: "$\\nu_0=10$",
        2: "$\\nu_0=15$"
    }

    for i in range(3):
        y_pos = start_y - i * (cbar_height + spacing)
        cbar_ax = fig.add_axes([box.x0, y_pos + 0.02, box.width, cbar_height])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick_label_dict[i])
        if i == 1:  # Label only the middle colorbar
            cbar.set_label(plot_cfg.plot.param_latex_names.estimatedKSDposteriors, labelpad=25)

        fig.text(
            box.x0 - 0.05,  # X position (left of colorbar)
            0.95 * y_pos + cbar_height / 2,  # Y position (vertical center of bar)
            text_labels[i],
            ha='right', va='center',
            fontsize=plot_cfg.plot.font.size
        )

    # Save
    output_path = os.path.join(output_dir, "toy_inverse_wishart.pdf")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved Inverse Wishart scale ellipse plot to: {output_path}")


def plot_sdp_comparisons_multiple_radii(
    basis_function,
    psi_sdp_list: list[np.ndarray],
    radius_labels: list[float],
    ksd_estimates: list[float],
    prior_distribution,
    plot_cfg: DictConfig,
    output_dir: str,
    domain: tuple = (-5, 5),
    resolution: int = 200,
) -> None:
    """
    Plot log prior and its SDP-relaxed approximations for multiple radius lower bounds.
    Saves a single-panel figure to PDF.

    Notes:
        - Only 1D plotting is supported (basis on x ∈ R).
        - prior_samples / posterior_samples are accepted for backwards-compat, but unused.
    """
    os.makedirs(output_dir, exist_ok=True)

    # LaTeX + fonts (align with your other plots)
    plt.rcParams.update({
        "font.size": plot_cfg.plot.font.size,
        "font.family": plot_cfg.plot.font.family,
        "text.usetex": plot_cfg.plot.font.use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    # Figure
    fig, ax = plt.subplots(
        figsize=(
            plot_cfg.plot.figure.size.width*1.3,
            plot_cfg.plot.figure.size.height,
        ),
        dpi=plot_cfg.plot.figure.dpi,
    )

    # Grid on x and target log prior
    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    phi_x = basis_function.evaluate(x)
    log_prior = prior_distribution.log_pdf(x).flatten()

    # Colors (cycle through palette for each radius curve)
    palette = list(getattr(plot_cfg.plot.color_palette, "colors", []))
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]  # matplotlib defaults fallback

    # Plot SDP approximations (shifted by best constant to match mean on grid)
    names = plot_cfg.plot.param_latex_names
    ksd_label = names.get("estimatedKSDposteriors", r"$\widehat{\mathrm{KSD}}^2$")
    approx_sym = r"$\approx$"
    geq_sym = r"$\geq$"

    for i, (psi, r_label, ksd) in enumerate(zip(psi_sdp_list, radius_labels, ksd_estimates)):
        f_sdp = (phi_x @ psi).flatten()
        c = float(np.mean(log_prior - f_sdp))
        color = palette[i % len(palette)]
        label = rf"r {geq_sym} {r_label} ({approx_sym} {ksd:.2f})"

        ax.plot(
            x.flatten(),
            f_sdp + c,
            label=label,
            linewidth=1.5,
            color=color,
        )

    # Plot base (true) log prior
    ax.plot(
        x.flatten(),
        log_prior,
        label=names.get("logbaseprior", "log Base Log Prior"),
        linestyle="--",
        linewidth=1.5,
        color="black",
    )

    # Labels and styling from latex name map
    ax.set_xlabel(names.get("theta", "theta"))

    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(
        title=ksd_label,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        handlelength=2,
        handletextpad=0.8,
        borderpad=0.6,
    )

    # Align legend title with labels
    leg.get_title().set_ha("right")
    leg._legend_box.align = "right"

    plt.setp(leg.get_texts(), fontsize=plt.rcParams["font.size"] * 0.9)
    plt.setp(leg.get_title(), fontsize=plt.rcParams["font.size"] * 0.95)

    if getattr(plot_cfg.plot.figure, "tight_layout", False):
        plt.tight_layout()

    # Save and close
    filename = "toy_gaussian_model_nonparametric_optimisation.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved SDP log prior comparison plot: {save_path}")


def plot_sdp_vs_ksd_minimizers(
    basis_function,
    psi_sdp_list: list[np.ndarray],
    psi_ksd_list: list[np.ndarray],
    radius_labels: list[float],
    ksd_estimates_sdp: list[float],
    ksd_estimates_ksd: list[float],
    prior_distribution,
    domain: tuple = (-5, 5),
    resolution: int = 200,
):
    """
    Plot log prior approximations: SDP vs KSD-minimizing solutions for multiple radii.

    Args:
        basis_function: Instance of BaseBasisFunction.
        psi_sdp_list (list[np.ndarray]): ψ vectors from SDP.
        psi_ksd_list (list[np.ndarray]): ψ vectors from KSD minimization.
        radius_labels (list[float]): Radii used for each optimization.
        ksd_estimates_sdp (list[float]): KSD from SDP solutions.
        ksd_estimates_ksd (list[float]): KSD from KSD-minimizing solutions.
        prior_distribution: Prior with .log_pdf(x).
        domain (tuple): Plotting domain.
        resolution (int): Number of x-points.
    """
    x = np.linspace(domain[0], domain[1], resolution)[:, None]
    phi_x = basis_function.evaluate(x)
    log_prior = prior_distribution.log_pdf(x).flatten()

    plt.figure(figsize=(10, 6))

    for psi_sdp, psi_ksd, r_label, ksd_sdp, ksd_ksd in zip(
        psi_sdp_list, psi_ksd_list, radius_labels, ksd_estimates_sdp, ksd_estimates_ksd
    ):
        f_sdp = phi_x @ psi_sdp
        f_ksd = phi_x @ psi_ksd

        # plt.plot(x, f_sdp, label=f"SDP (r ≥ {r_label}) | KSD ≈ {ksd_sdp:.2e}", linestyle="-")
        plt.plot(x, f_ksd, label=f"KSD-min (r = {r_label}) | KSD ≈ {ksd_ksd:.2e}", linestyle="--")

    plt.plot(x, log_prior, label="True Log Prior", color="black", linewidth=2, linestyle=":")

    plt.title("SDP vs KSD-Minimizing Log Prior Approximations")
    plt.xlabel("x")
    plt.ylabel("Log Prior Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
