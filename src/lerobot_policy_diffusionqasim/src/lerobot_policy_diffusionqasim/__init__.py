# __init__.py
"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_diffusionqasim import DiffusionQasimConfig
from .modeling_diffusionqasim import DiffusionQasimPolicy
from .processor_diffusionqasim import make_diffusionqasim_pre_post_processors

__all__ = [
    "DiffusionQasimConfig",
    "DiffusionQasimPolicy",
    "make_diffusionqasim_pre_post_processors",
]