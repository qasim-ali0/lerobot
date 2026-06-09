# modeling_diffusionqasim.py
import torch
import torch.nn as nn
from typing import Any
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from collections import deque

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGES
from .configuration_diffusionqasim import DiffusionQasimConfig
from .blocks import *

class DiffusionQasimPolicy(PreTrainedPolicy):
    config_class = DiffusionQasimConfig
    name = "diffusionqasim"

    def __init__(self, config: DiffusionQasimConfig, dataset_stats: dict[str, Any] = None, **kwargs):
        super().__init__(config, dataset_stats)
        config.validate_features()  # not called automatically by the base class
        self.config = config
        backbone_model = torchvision.models.resnet18(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights="ResNet18_Weights.IMAGENET1K_V1",
            norm_layer=FrozenBatchNorm2d,
        )
        self.vision_backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

        self.img_feat_proj = nn.Linear(512, 512)
        self.action_in_proj = nn.Linear(6, 512)
        self.state_proj = nn.Linear(6, 512)
        self.action_out_proj = nn.Linear(512, 6)
        self.time_emb = step_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(128),        # dsed = 256
            nn.Linear(128, 128 * 4),     # 256 → 1024
            nn.Mish(),
            nn.Linear(128 * 4, 128),     # 1024 → 256
        )
        self.T = 16
        self.convblocks = nn.ModuleList([
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock()
        ])
        self.spatial_softmax = SpatialSoftmax(512, feature_dim=512)
        
        
    def reset(self):
        """Reset episode state."""
        self._action_queue.clear()

    def get_optim_params(self):
        """Return parameters to pass to the optimizer (e.g. with per-group lr/wd)."""
        return self.parameters()

    def predict_action_chunk(self, batch):
        B = batch[OBS_STATE].shape[0]
        state = self.state_proj(batch[OBS_STATE])
        img_embed = self.img_feat_proj(self.spatial_softmax(
            self.vision_backbone(batch[next(iter(self.config.image_features))])["feature_map"]))

        x = torch.randn(B, self.config.chunk_size, 6, device=state.device)  # t=1: pure noise
        dt = 1.0 / self.T 
        for i in reversed(range(self.T)):                                   # integrate t: 1 → 0
            t = torch.full((B, 1), (i + 1) * dt, device=state.device)
            a = self.action_in_proj(x)
            t_emb = self.time_emb(t * 1000)
            for m in self.convblocks:
                a = m(a, state, t_emb, img_embed)
            v = self.action_out_proj(a)                                     # velocity = noise - action
            x = x - dt * v                                                  # Euler step toward data
        return x

    def select_action(self, batch, **kwargs):
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]  # (B, n_action_steps, 6)
            # transpose so iterating yields n_action_steps tensors of (B, 6)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()   # (B, 6)
    
    def forward(self, batch: dict[str, torch.Tensor], use_mean=False) -> tuple[torch.Tensor, dict]:
        """Compute the training loss.

        `batch["action_is_pad"]` is a bool mask of shape (B, horizon) that marks
        timesteps padded because the episode ended before `horizon` steps, you
        can exclude those from your loss.
        """
        # 'action', 'next.reward', 'next.done', 'next.truncated', 'info', 'action_is_pad', 'task', 'index', 'task_index', 'episode_index', 'timestamp', 'observation.images.front', 'observation.state'
        B = batch[OBS_STATE].shape[0]
        t = torch.rand(B, 1, device=batch[OBS_STATE].device)   # (B, 1)
        noise = torch.randn_like(batch[ACTION])
        tb = t[..., None]                                       # (B, 1, 1)
        actions_noised = tb * noise + (1 - tb) * batch[ACTION]
        actions = self.action_in_proj(actions_noised)
        
        state = self.state_proj(batch[OBS_STATE])
        images = [batch[key] for key in self.config.image_features]
        # then pass to backbone
        features = []
        for img in images:
            features.append(self.vision_backbone(img)["feature_map"])  # (B, 512, 15, 20)
        img_embed = self.spatial_softmax(features[0])
        img_embed = self.img_feat_proj(img_embed)

        t_emb = self.time_emb(t*1000)                     # (B, 512)

        for m in self.convblocks:
            actions = m(actions, state, t_emb, img_embed)
        
        pred_noise = self.action_out_proj(actions)

        l2_loss = torch.nn.functional.mse_loss(pred_noise, noise-batch[ACTION], reduction="none")
        valid = ~batch["action_is_pad"].unsqueeze(-1)  # (B, T, 1)
        den = 6 * valid.sum()
        l2_loss = (l2_loss * valid).sum() / den
        
        loss = l2_loss 
        
        return loss, {
            'l2_loss': l2_loss.item(),
            'loss': loss.item()
        }
    
class ConvBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_gamma_scale = nn.Sequential(
            nn.Linear(512+512+128, 512 * 2),
            nn.Mish(),
            nn.Linear(512*2, 512 * 2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=5, padding='same'),
            nn.GroupNorm(64, 512),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=5, padding='same'),
            nn.GroupNorm(64, 512),
            nn.Mish(),
        )
        self.norm = nn.LayerNorm(512)

    def forward(self, actions, state, timestep, img_embed):
        # conv path works in (B, C, L)
        x = actions.transpose(1, 2)                 # (B, 512, L)
        x = x + self.conv1(x)
        actions = x.transpose(1, 2)                 # back to (B, L, 512)

        # FiLM / LayerNorm path works in (B, L, C)
        cond = torch.concat([state, timestep, img_embed], -1)
        gamma, scale = self.mlp_gamma_scale(cond).chunk(2, -1)
        actions = actions + self.norm(actions) * scale[:, None] + gamma[:, None]

        x = actions.transpose(1, 2)
        x = x + self.conv2(x)
        return x.transpose(1, 2)
        
class SpatialSoftmax(nn.Module):
    def __init__(self, in_channels, num_kp=32, temperature=1.0, feature_dim=64):
        super().__init__()
        self.num_kp = num_kp
        self.temperature = temperature

        # project 512 channels → num_kp "keypoint heatmap" channels
        self.kp_conv = nn.Conv2d(in_channels, num_kp, kernel_size=1)

        # optional linear projection after extracting coords
        self.fc = nn.Sequential(
            nn.Linear(num_kp * 2, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # project to keypoint channels
        h = self.kp_conv(x)                           # (B, num_kp, H, W)

        # softmax over spatial dimensions
        h = h.view(B, self.num_kp, -1)                # (B, num_kp, H*W)
        h = F.softmax(h / self.temperature, dim=-1)   # (B, num_kp, H*W)

        # build normalized coordinate grid [-1, 1]
        lin_y = torch.linspace(-1, 1, H, device=x.device)
        lin_x = torch.linspace(-1, 1, W, device=x.device)
        grid_y, grid_x = torch.meshgrid(lin_y, lin_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        grid = grid.view(-1, 2)                        # (H*W, 2)

        # expected keypoint coordinates
        # h: (B, num_kp, H*W), grid: (H*W, 2)
        kp = torch.einsum('bki,id->bkd', h, grid)     # (B, num_kp, 2)
        kp = kp.view(B, -1)                            # (B, num_kp*2)

        return self.fc(kp)