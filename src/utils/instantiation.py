import importlib
from typing import Any
from typing import Dict


def get_class_from_path(class_path: str) -> Any:
    """
    Dynamically import a class from a string path.

    Args:
        class_path (str): Full class path, e.g. "src.distributions.gaussian.Gaussian"

    Returns:
        Any: Imported class.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate_distribution(distribution_config: Dict[str, Any]) -> Any:
    """Instantiate a distribution class from a config.

    Args:
        distribution_config (Dict[str, Any]): Dictionary with 'distribution' and optional 'parameters'.

    Returns:
        Any: Instantiated distribution object.
    """
    cls = get_class_from_path(distribution_config['distribution'])
    params = distribution_config.get('parameters', {})
    return cls(**params)