from typing import Dict, Any
import yaml
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_plot_config(path: str):
    with open(path, "r") as f:
        plot_cfg = yaml.safe_load(f)
    return OmegaConf.create(plot_cfg)  # optional: convert to DictConfig for consistency
