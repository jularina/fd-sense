from __future__ import annotations
import importlib
import inspect
import pkgutil
from typing import Dict, List, Optional, Type, Any

from src.utils.distributions import DISTRIBUTION_MAP
from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY

def _optional_attr(modpath: str, attr: str):
    try:
        mod = importlib.import_module(modpath)
        return getattr(mod, attr, None)
    except Exception:
        return None

KERNELS_REGISTRY = _optional_attr("src.kernels.registry", "KERNELS_REGISTRY")
LOSS_REGISTRY    = _optional_attr("src.losses.registry", "LOSS_REGISTRY")

BaseKernel = _optional_attr("src.kernels.base", "BaseKernel")
BaseBasis  = _optional_attr("src.basis_functions.base", "BaseBasisFunction")
BaseLoss   = _optional_attr("src.losses.base", "BaseLoss")

def _walk_concrete_subclasses(package_name: str, base_cls: Optional[type]) -> List[Type]:
    if base_cls is None:
        return []
    try:
        pkg = importlib.import_module(package_name)
    except Exception:
        return []
    if not hasattr(pkg, "__path__"):
        return []
    classes: List[Type] = []
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            try:
                if issubclass(obj, base_cls) and obj is not base_cls and not inspect.isabstract(obj):
                    classes.append(obj)
            except Exception:
                continue
    # dedupe
    seen, out = set(), []
    for c in classes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _qualname(obj: Any) -> str:
    try:
        return f"{obj.__module__}.{obj.__name__}"
    except Exception:
        return repr(obj)

def _fmt_sig(cls: Type) -> str:
    try:
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        return f"{cls.__name__}({', '.join(str(p) for p in params)})"
    except Exception:
        return f"{cls.__name__}(...)"

class PlaygroundZoo:
    """Runtime registry/inspector of supported components."""

    # ---- Distributions (registry-backed)
    @staticmethod
    def list_distributions() -> List[str]:
        return [f"{name}  ->  {_qualname(cls)}" for name, cls in DISTRIBUTION_MAP.items()]

    # ---- Kernels: prefer registry, else discover subclasses
    @staticmethod
    def list_kernels() -> List[str]:
        if isinstance(KERNELS_REGISTRY, dict) and KERNELS_REGISTRY:
            # Assume {name: class or factory}
            out = []
            for name, obj in KERNELS_REGISTRY.items():
                out.append(f"{name}  ->  {_qualname(obj)}")
            return out
        # Fallback: discovery
        classes = _walk_concrete_subclasses("src.kernels", BaseKernel)
        return [_fmt_sig(c) for c in classes]

    # ---- Basis functions: registry-backed
    @staticmethod
    def list_basis_functions() -> List[str]:
        # Assume {name: class or callable}
        out = []
        for name, obj in BASIS_FUNCTIONS_REGISTRY.items():
            out.append(f"{name}  ->  {_qualname(obj)}")
        # If you also want constructor sigs and these are classes:
        # out = [f"{name}  ->  {_fmt_sig(obj) if inspect.isclass(obj) else _qualname(obj)}" for name, obj in BASIS_FUNCTIONS_REGISTRY.items()]
        return out

    # ---- Losses: prefer registry, else discover subclasses
    @staticmethod
    def list_losses() -> List[str]:
        if isinstance(LOSS_REGISTRY, dict) and LOSS_REGISTRY:
            return [f"{name}  ->  {_qualname(obj)}" for name, obj in LOSS_REGISTRY.items()]
        classes = _walk_concrete_subclasses("src.losses", BaseLoss)
        return [_fmt_sig(c) for c in classes]

    # ---- Pretty printers / exports
    @classmethod
    def show_all(cls) -> None:
        sections = [
            ("Distributions", cls.list_distributions()),
            ("Kernels", cls.list_kernels()),
            ("Basis Functions", cls.list_basis_functions()),
            ("Losses / Objectives", cls.list_losses()),
        ]
        for title, items in sections:
            print(f"\n=== {title} ===")
            if items:
                for it in items:
                    print(f"- {it}")
            else:
                print("(none found)")

    @classmethod
    def to_dict(cls) -> Dict[str, List[str]]:
        return {
            "Distributions": cls.list_distributions(),
            "Kernels": cls.list_kernels(),
            "Basis Functions": cls.list_basis_functions(),
            "Losses / Objectives": cls.list_losses(),
        }

    # ---- Optional: config validation helper
    @classmethod
    def validate_config(cls, cfg) -> None:
        """
        Quickly sanity-check likely _target_ strings against discovered/registered classes.
        Prints warnings instead of raising, to keep it user-friendly.
        """
        known = set()
        # from registries
        for _, cls_ in DISTRIBUTION_MAP.items(): known.add(_qualname(cls_))
        for reg in (KERNELS_REGISTRY, BASIS_FUNCTIONS_REGISTRY, LOSS_REGISTRY):
            if isinstance(reg, dict):
                for _, obj in reg.items():
                    known.add(_qualname(obj))
        # from discovery
        for c in _walk_concrete_subclasses("src.kernels", BaseKernel): known.add(_qualname(c))
        for c in _walk_concrete_subclasses("src.losses", BaseLoss): known.add(_qualname(c))

        def _check(path: str, node):
            try:
                target = node.get("_target_")
            except Exception:
                target = None
            if target and target not in known:
                print(f"[Zoo] Warning: {path}._target_={target!r} not recognized by the zoo.")

        # common spots
        try:
            _check("cfg.model", cfg.model)
        except Exception:
            pass
        try:
            _check("cfg.ksd.kernel", cfg.ksd.kernel)
        except Exception:
            pass
        try:
            _check("cfg.ksd.optimize.prior.nonparametric", cfg.ksd.optimize.prior.nonparametric)
        except Exception:
            pass

if __name__ == "__main__":
    PlaygroundZoo.show_all()
    catalog = PlaygroundZoo.to_dict()