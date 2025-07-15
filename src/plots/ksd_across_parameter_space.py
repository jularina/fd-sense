import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from omegaconf import DictConfig
from matplotlib.colors import LinearSegmentedColormap
import colorsys
from matplotlib.colors import to_rgb


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
            fig.savefig(save_path, format="pdf")
            plt.close(fig)

            print(f"Saved heatmap to: {save_path}")


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

                ax.plot(x_vals, y_vals, marker='o', color=plot_cfg.plot.color_palette.colors[0])

                ax.set_xlabel(varying_param_latex)
                ylabel = "log KSD" if plot_cfg.plot.y_axis.log_scale else "KSD"
                ax.set_ylabel(ylabel)
                ax.set_title(f"KSD vs {varying_param_latex} (fixed {fixed_param_latex} = {fixed_val:.1f})")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                if plot_cfg.plot.figure.tight_layout:
                    plt.tight_layout()

                if plot_cfg.plot.y_axis.log_scale:
                    ax.set_yscale("log")

                filename = f"ksd_line_{fixed_param_name}_{fixed_val:.2f}_vs_{varying_param_name}.pdf"
                save_path = os.path.join(output_dir, filename)
                fig.savefig(save_path, format="pdf")
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
        norm = (avg_ksd_per_line - np.min(avg_ksd_per_line)) / (
                    np.max(avg_ksd_per_line) - np.min(avg_ksd_per_line) + 1e-12)
        brightness_vals = 0.3 + 0.7 * (1 - norm)  # range [0.3, 1.0]

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
                marker='o',
                label=f"{fixed_param_latex} = {fixed_val:.2f}",
                color=shaded_rgb,
            )

        ax.set_xlabel(varying_param_latex)
        ylabel = "log KSD" if plot_cfg.plot.y_axis.log_scale else "KSD"
        ax.set_ylabel(ylabel)
        ax.set_title(f"KSD vs {varying_param_latex} (multiple {fixed_param_latex})")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

        if plot_cfg.plot.y_axis.log_scale:
            ax.set_yscale("log")

        if plot_cfg.plot.figure.tight_layout:
            plt.tight_layout()

        save_path = os.path.join(output_dir, f"{filename_prefix}_{varying_param_name}_vs_{fixed_param_name}.pdf")
        fig.savefig(save_path, format="pdf")
        plt.close(fig)
        print(f"Saved combined KSD vs {varying_param_name} plot to: {save_path}")

    # Plot 1: Vary param1, lines for each fixed param2
    make_multi_line_plot(fixed_idx=1, varying_idx=0, filename_prefix="ksd_multiline")

    # Plot 2: Vary param2, lines for each fixed param1
    make_multi_line_plot(fixed_idx=0, varying_idx=1, filename_prefix="ksd_multiline")
