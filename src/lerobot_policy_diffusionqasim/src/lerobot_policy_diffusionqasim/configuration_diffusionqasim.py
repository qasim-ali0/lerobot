# configuration_diffusionqasim.py
from dataclasses import dataclass, field
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.configs import NormalizationMode, PreTrainedConfig

@PreTrainedConfig.register_subclass("diffusionqasim")
@dataclass
class DiffusionQasimConfig(PreTrainedConfig):
    """Configuration class for DiffusionQasimPolicy.

    Args:
        n_obs_steps: Number of observation steps to use as input
        horizon: Action prediction horizon
        n_action_steps: Number of action steps to execute
        hidden_dim: Hidden dimension for the policy network
        # Add your policy-specific parameters here
    """
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE":  NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    # horizon: int = 50
    # n_action_steps: int = 50
    # hidden_dim: int = 256
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 16
    replace_final_stride_with_dilation: bool = False

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()


    def validate_features(self) -> None:
        """Validate input/output feature compatibility."""
        if not self.image_features:
            raise ValueError("DiffusionQasimPolicy requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("DiffusionQasimPolicy requires 'action' in output_features.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Relative timestep offsets the dataset loader provides per observation.

        Return `None` for single-frame policies. For temporal policies that consume
        multiple past or future frames, return a list of offsets, e.g. `[-20, -10, 0, 10]` for
        3 past frames at stride 10 and 1 future frame at stride 10.
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Relative timestep offsets for the action chunk the dataset loader returns.
        """
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None