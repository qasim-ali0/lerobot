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
        self.state_proj = nn.Linear(6, 512)
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(128),        # dsed = 256
            nn.Linear(128, 128 * 4),     # 256 → 1024
            nn.Mish(),
            nn.Linear(128 * 4, 128),     # 1024 → 256
        )
        self.T = 16
        self.unet = ConditionalUNet1d()
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
            t_emb = self.time_emb(t * 1000)
            cond = torch.concat([state, t_emb, img_embed], -1)             # (B, 1152)
            v = self.unet(x, cond)                                          # velocity = noise - action
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
        actions_noised = tb * noise + (1 - tb) * batch[ACTION]  # (B, L, 6)

        state = self.state_proj(batch[OBS_STATE])
        images = [batch[key] for key in self.config.image_features]
        # then pass to backbone
        features = []
        for img in images:
            features.append(self.vision_backbone(img)["feature_map"])  # (B, 512, 15, 20)
        img_embed = self.spatial_softmax(features[0])
        img_embed = self.img_feat_proj(img_embed)

        t_emb = self.time_emb(t*1000)                     # (B, 128)
        cond = torch.concat([state, t_emb, img_embed], -1)  # (B, 1152)

        pred_noise = self.unet(actions_noised, cond)      # (B, L, 6)

        l2_loss = torch.nn.functional.mse_loss(pred_noise, noise-batch[ACTION], reduction="none")
        valid = ~batch["action_is_pad"].unsqueeze(-1)  # (B, T, 1)
        den = 6 * valid.sum()
        l2_loss = (l2_loss * valid).sum() / den
        
        loss = l2_loss 
        
        return loss, {
            'l2_loss': l2_loss.item(),
            'loss': loss.item()
        }
    
class ResidualBlock1d(nn.Module):
    """Conv -> FiLM -> Conv, with a residual skip. Works in (B, C, L)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out_ch = out_ch
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding='same'),
            nn.GroupNorm(8, out_ch),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding='same'),
            nn.GroupNorm(8, out_ch),
            nn.Mish(),
        )
        # FiLM: cond = [state, timestep, img_embed] = 512 + 128 + 512 = 1152
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(512+128+512, out_ch * 2),
        )
        self.residual_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        # x: (B, C, L), cond: (B, 1152)
        out = self.conv1(x)
        gamma, beta = self.cond_encoder(cond).unsqueeze(-1).chunk(2, dim=1)  # (B, out_ch, 1) each
        out = gamma * out + beta
        out = self.conv2(out)
        return out + self.residual_conv(x)


class ConditionalUNet1d(nn.Module):
    """1D UNet over the action-time axis with FiLM conditioning and skip connections."""
    def __init__(self):
        super().__init__()
        dims = [6, 256, 512, 1024]
        in_out = list(zip(dims[:-1], dims[1:]))  # [(6,256), (256,512), (512,1024)]

        # encoder: each level = 2 res blocks + a stride-2 downsample (except the last level)
        self.down_modules = nn.ModuleList([])
        for i, (din, dout) in enumerate(in_out):
            is_last = i >= len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ResidualBlock1d(din, dout),
                ResidualBlock1d(dout, dout),
                nn.Conv1d(dout, dout, 3, 2, 1) if not is_last else nn.Identity(),
            ]))

        self.mid_modules = nn.ModuleList([
            ResidualBlock1d(1024, 1024),
            ResidualBlock1d(1024, 1024),
        ])

        # decoder: takes the encoder skip (hence din*2) + a stride-2 upsample
        self.up_modules = nn.ModuleList([])
        for i, (dout, din) in enumerate(reversed(in_out[1:])):  # [(512,1024), (256,512)]
            is_last = i >= len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                ResidualBlock1d(din * 2, dout),
                ResidualBlock1d(dout, dout),
                nn.ConvTranspose1d(dout, dout, 4, 2, 1) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, padding='same'),
            nn.GroupNorm(8, 256),
            nn.Mish(),
            nn.Conv1d(256, 6, 1),
        )

    def forward(self, x, cond):
        # x: (B, L, 6), cond: (B, 1152)
        x = x.transpose(1, 2)                       # (B, 6, L)
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            skips.append(x)
            x = downsample(x)

        for m in self.mid_modules:
            x = m(x, cond)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)  # concat encoder skip on channels
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)                     # (B, L, 6)


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