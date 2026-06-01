from __future__ import annotations

from typing import TYPE_CHECKING

from .config import LatentFactorRouterConfig

__all__ = [
    "LatentFactorRouter",
    "LatentFactorTrainer",
    "LatentFactorRouterLiteLLM",
    "LatentFactorRouterConfig",
]

if TYPE_CHECKING:
    from .latent_factor_router import LatentFactorRouterLiteLLM
    from .router import LatentFactorRouter
    from .trainer import LatentFactorTrainer


def __getattr__(name: str):
    if name == "LatentFactorRouter":
        try:
            from .router import LatentFactorRouter as exported
        except ImportError as exc:
            raise ImportError(
                "LatentFactorRouter requires the optional superclaw_routers extras. "
                "Install them with `pip install litellm[superclaw_routers]`."
            ) from exc
        return exported

    if name == "LatentFactorTrainer":
        try:
            from .trainer import LatentFactorTrainer as exported
        except ImportError as exc:
            raise ImportError(
                "LatentFactorTrainer requires the optional superclaw_routers extras. "
                "Install them with `pip install litellm[superclaw_routers]`."
            ) from exc
        return exported

    if name == "LatentFactorRouterLiteLLM":
        try:
            from .latent_factor_router import LatentFactorRouterLiteLLM as exported
        except ImportError as exc:
            raise ImportError(
                "LatentFactorRouterLiteLLM requires the optional superclaw_routers extras. "
                "Install them with `pip install litellm[superclaw_routers]`."
            ) from exc
        return exported

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
