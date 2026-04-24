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


def instantiate_from_target_str(target: str, kwargs: Dict[str, Any]):
    module_name, cls_name = target.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(**kwargs)


def deepcopy_cfg(cfg: DictConfig) -> DictConfig:
    # Safe deep copy preserving OmegaConf structure
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _to_serialisable(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    return obj


def save_to_serializable_json(results: Dict[str, Any], path: str) -> None:
    results_ser = _to_serialisable(results)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_ser, f, indent=2)


def load_results_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        results = json.load(f)
    return results


def convert_dim_keys_to_int(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert nested dimension keys like "5" back to integers in a loaded results dict.
    """
    for top_key in ["error_mean", "error_ci", "time_mean", "time_ci"]:
        if top_key not in results:
            continue

        for method, method_dict in results[top_key].items():
            if isinstance(method_dict, dict):
                results[top_key][method] = {
                    int(k): v for k, v in method_dict.items()
                }

    return results
