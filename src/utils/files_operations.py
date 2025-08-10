from typing import Dict, Any
import yaml
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import os
import csv
import json


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_plot_config(path: str):
    with open(path, "r") as f:
        plot_cfg = yaml.safe_load(f)
    return OmegaConf.create(plot_cfg)

def get_outdir(cfg: DictConfig) -> str:
    output_dir = os.path.join(get_original_cwd(), cfg.flags.plots.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_json(obj: Any, path: str) -> None:
    def _default(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def save_csv(rows, fieldnames, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r[k].item() if isinstance(r[k], (np.floating, np.integer)) else r[k]) for k in fieldnames})


def load_numpy_array(path: str) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Expected numpy array in {path}")
    return arr
