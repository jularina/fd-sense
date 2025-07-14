import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List


def plot_ksd_heatmaps(
    ksd_results: Dict[Tuple[float, ...], float],
    param_names: List[str],
) -> None:
    """
    Plots heatmaps of KSD values for each pair of parameters.

    Parameters:
        ksd_results (Dict[Tuple[float, ...], float]): A dictionary where the keys are tuples of parameter values
            (in the same order as param_names), and the values are the corresponding KSD values.
        param_names (List[str]): The list of parameter names in the order used in the keys of ksd_results.

    Returns:
        None: Displays heatmaps for each unique pair of parameters.
    """
    num_params = len(param_names)
    param_values = np.array(list(ksd_results.keys()))
    ksd_values = np.array(list(ksd_results.values()))

    for i in range(num_params):
        for j in range(i + 1, num_params):
            x_vals = param_values[:, i]
            y_vals = param_values[:, j]
            ksd_vals = ksd_values

            # Create a grid for plotting
            xi = np.unique(x_vals)
            yi = np.unique(y_vals)
            zi = np.full((len(yi), len(xi)), np.nan)

            for x, y, ksd in zip(x_vals, y_vals, ksd_vals):
                x_idx = np.where(xi == x)[0][0]
                y_idx = np.where(yi == y)[0][0]
                zi[y_idx, x_idx] = ksd

            plt.figure(figsize=(6, 5))
            sns.heatmap(zi, xticklabels=np.round(xi, 3), yticklabels=np.round(yi, 3),
                        cmap="viridis", annot=True, fmt=".3f")
            plt.xlabel(param_names[i])
            plt.ylabel(param_names[j])
            plt.title(f"KSD heatmap: {param_names[i]} vs {param_names[j]}")
            plt.tight_layout()
            plt.show()