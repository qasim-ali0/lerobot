# modeling_actqasim.py
import torch
import torch.nn as nn
from typing import Any
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from collections import deque

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGES
from .configuration_actqasim import ActQasimConfig
from .blocks import *

class ActQasimPolicy(PreTrainedPolicy):
    config_class = ActQasimConfig
    name = "actqasim"

    def __init__(self, config: ActQasimConfig, dataset_stats: dict[str, Any] = None, **kwargs):
        super().__init__(config, dataset_stats)
        config.validate_features()  # not called automatically by the base class
        self.config = config
        backbone_model = torchvision.models.resnet18(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights="ResNet18_Weights.IMAGENET1K_V1",
            norm_layer=FrozenBatchNorm2d,
        )
        self.vision_backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        
        self.cvae = CVAE()
        self.img_feat_proj = nn.Linear(512, 512)
        self.transformer_encoder = nn.Sequential(
            
                TransformerLayer(512, 8, 1024),
                TransformerLayer(512, 8, 1024),
                TransformerLayer(512, 8, 1024),
                TransformerLayer(512, 8, 1024),
            )
        self.transformer_decoder = nn.ModuleList([
            TransformerLayer(512, 8, 1024, cross_attention=True) for _ in range(7)
        ])
        self.action_embeds = nn.Parameter(torch.randn(self.config.chunk_size, 512))
        self.action_in_proj = nn.Linear(6, 512)
        self.state_proj = nn.Linear(6, 512)
        self.z_proj = nn.Linear(32, 512)
        self.action_out_proj = nn.Linear(512, 6)
        self.temporal_ensembler = ACTTemporalEnsembler(0.01, config.chunk_size)
        self.reset()
        
    def reset(self):
        """Reset episode state."""
        self.temporal_ensembler.reset()
            
    def get_optim_params(self):
        """Return parameters to pass to the optimizer (e.g. with per-group lr/wd)."""
        return self.parameters()

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return the full action chunk (B, chunk_size, action_dim) for the current observation."""
        B = batch[OBS_STATE].shape[0]
        state = self.state_proj(batch[OBS_STATE])
        images = [batch[key] for key in self.config.image_features]

        features = []
        for img in images:
            features.append(self.vision_backbone(img)["feature_map"])  # (B, 512, 15, 20)
        img_feats = torch.flatten(features[0], 2).transpose(1, 2)  # (B, 300, 512)
        img_feats = self.img_feat_proj(img_feats)

        z = torch.zeros(B, 32, device=state.device, dtype=state.dtype)
        x = torch.concat([img_feats, state[:, None], self.z_proj(z)[:, None]], 1)

        x = self.transformer_encoder(x)
        action_embeds = self.action_embeds.unsqueeze(0).expand(B, -1, -1)
        for layer in self.transformer_decoder:
            action_embeds = layer(action_embeds, context=x)
        return self.action_out_proj(action_embeds)  # (B, chunk_size, action_dim)

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return a single action for the current timestep (called at inference)."""
        
        action_chunk = self.predict_action_chunk(batch)
        action = self.temporal_ensembler.update(action_chunk)
        return action

    def forward(self, batch: dict[str, torch.Tensor], use_mean=False) -> tuple[torch.Tensor, dict]:
        """Compute the training loss.

        `batch["action_is_pad"]` is a bool mask of shape (B, horizon) that marks
        timesteps padded because the episode ended before `horizon` steps, you
        can exclude those from your loss.
        """
        # 'action', 'next.reward', 'next.done', 'next.truncated', 'info', 'action_is_pad', 'task', 'index', 'task_index', 'episode_index', 'timestamp', 'observation.images.front', 'observation.state'
        B = batch[OBS_STATE].shape[0]
        actions = self.action_in_proj(batch[ACTION])
        state = self.state_proj(batch[OBS_STATE])
        images = [batch[key] for key in self.config.image_features]

        # then pass to backbone
        features = []
        for img in images:
            features.append(self.vision_backbone(img)["feature_map"])  # (B, 512, 15, 20)
        img_feats = torch.flatten(features[0], 2).transpose(1, 2)
        assert img_feats.shape[1:] == (300, 512)
        img_feats = self.img_feat_proj(img_feats)
        
        z_mean, log_sigma_x2_hat = self.cvae(state, actions)
        if not use_mean:
            z = torch.distributions.Normal(z_mean, log_sigma_x2_hat.div(2).exp()).rsample()
        else:
            z= z_mean
        x = torch.concat([img_feats, state[:, None], self.z_proj(z)[:, None]], 1)
        
        x = self.transformer_encoder(x)
        action_embeds = self.action_embeds.unsqueeze(0).expand(B, -1, -1)
        for layer in self.transformer_decoder:
            action_embeds = layer(action_embeds, context=x)
        actions_out = self.action_out_proj(action_embeds)

        l1_loss = torch.nn.functional.l1_loss(actions_out, batch[ACTION], reduction="none")
        valid = ~batch["action_is_pad"].unsqueeze(-1)  # (B, T, 1)
        den = 6 * valid.sum()
        l1_loss = (l1_loss * valid).sum() / den
        
        mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - z_mean.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
        loss = l1_loss + 10.0 * mean_kld
        
        return loss, {
            'l1_loss': l1_loss.item(),
            'reg': mean_kld.item(),
            'loss': loss.item()
        }
    
class CVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(torch.randn(512), True)
        self.transformer_layers  = nn.Sequential(
            TransformerLayer(512, 8, 1024),
            TransformerLayer(512, 8, 1024),
            TransformerLayer(512, 8, 1024),
            TransformerLayer(512, 8, 1024),
        )
        self.final_proj = nn.Linear(512, 64)
        # self.register_buffer("cls_token", cls_token)
    
    def forward(self, state, actions):
        B = state.shape[0]
        x = torch.concat([self.cls_token[None, None].expand(B, 1, -1), state[:, None], actions], 1)
        x = self.transformer_layers(x)
        z_mean, log_sigma_x2_hat = torch.split(self.final_proj(x[:, 0]), 32, 1)
        return z_mean, log_sigma_x2_hat
        
class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()
        self.absolute_time = 0
    
    def reset(self):
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
    
    def update(self, action_chunk):
        if self.absolute_time == 0:
            self.ensembled_actions = action_chunk[:, 1:].clone()
            self.ensembled_actions_count = torch.ones((self.chunk_size -1 , 1), dtype=torch.int32, device=action_chunk.device)
            action = action_chunk[:, 0]
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += self.ensemble_weights[self.ensembled_actions_count]*action_chunk[:, :-1]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count ]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size) 
            action = self.ensembled_actions[:, 0]
            self.ensembled_actions[:, :-1] = self.ensembled_actions[:, 1:]
            self.ensembled_actions[:, -1] = action_chunk[:, -1]
            self.ensembled_actions_count[:-1] = self.ensembled_actions_count[1:]
            self.ensembled_actions_count[-1] = 1
        self.absolute_time += 1
        return action 

        # self.ensembled_actions = 
        # self.ensembled_actions_sum[:, :-1] = self.total_actions[:, 1:] + 1
        # self.total_actions[:, -1] = 1
        # self.ensembled_actions_sum = self.ensemble_weights
        # self.ensembled_actions 
        # self.ensembled_actions_count += 
        # action = self.total_actions
        # self.total_actions[] = action_chunk.clone()
            
        # return action