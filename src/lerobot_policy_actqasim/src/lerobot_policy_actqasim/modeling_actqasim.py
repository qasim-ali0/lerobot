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



class AttentionEncoder(nn.Module):
    """Multi-head self-attention with RoPE, optionally cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RoPE2D(self.d_head)
        self.norm1 = nn.LayerNorm(d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, n_heads, T, d_head)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        img_feats: torch.Tensor, 
        state: torch.Tensor, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, T, D) query input
            context: (B, S, D) key/value source — if None, self-attention
            mask:    (B, 1, T, S) or broadcastable boolean mask (True = ignore)
        Returns:
            (B, T, D)
        """
        img_feats_og, state_og, z_og = img_feats, state, z
        img_feats, state, z = self.norm1(img_feats), self.norm1(state), self.norm1(z)
        B, T, _ = img_feats.shape
        q_img = self.q(img_feats)
        k_img = self.k(img_feats)
        v_img = self.v(img_feats)
        rows = (torch.arange(T, device=img_feats.device) // 20).float()  # (300,)
        cols = (torch.arange(T, device=img_feats.device) %  20).float()  # (300,)
        q_img = self.rope(self._split_heads(q_img), rows, cols)
        k_img = self.rope(self._split_heads(k_img), rows, cols)
        v_img = self._split_heads(self.v(img_feats))
        
        q_state = self.q(state)
        k_state = self.k(state)
        v_state = self.v(state)
        q_z = self.q(z)
        k_z = self.k(z)
        v_z = self.v(z)
        
        q = torch.concat([q_img, self._split_heads(q_state[:, None]), self._split_heads(q_z[:, None])], 2)
        k = torch.concat([k_img, self._split_heads(k_state[:, None]), self._split_heads(k_z[:, None])], 2)
        v = torch.concat([v_img, self._split_heads(v_state[:, None]), self._split_heads(v_z[:, None])], 2)
        B, _, T, _ = q.shape

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, H, T, S)

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = attn @ v                            # (B, H, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.out(out)
        img_feats, state, z = out[:, :-2], out[:, -2].squeeze(1), out[:, -1].squeeze(1)
        return img_feats_og + img_feats, state_og + state, z_og + z


class ActEncoderLayer(nn.Module):
    """Pre-norm transformer layer: self-attention + optional cross-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = AttentionEncoder(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        img_feats,
        state,
        z,
    ) -> torch.Tensor:
                
        img_feats, state, z = self.self_attn(img_feats, state, z)

        # FFN (pre-norm)
        img_feats = img_feats + self.ff(self.norm2(img_feats))
        state = state + self.ff(self.norm2(state))
        z = z + self.ff(self.norm2(z))
        return img_feats, state, z




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
                ActEncoderLayer(512, 8, 1576),
                ActEncoderLayer(512, 8, 1576),
                ActEncoderLayer(512, 8, 1576),
                ActEncoderLayer(512, 8, 1576),
            )
        self.transformer_decoder = nn.ModuleList([
            TransformerLayer(512, 8, 1576, cross_attention=True) for _ in range(2)
        ])
        self.action_embeds = nn.Parameter(torch.randn(self.config.chunk_size, 512))
        self.action_in_proj = nn.Linear(6, 512)
        self.state_proj = nn.Linear(6, 512)
        self.z_proj = nn.Linear(32, 512)
        self.action_out_proj = nn.Linear(512, 6)
        self.temporal_ensembler = ACTTemporalEnsembler(0.00, config.chunk_size)
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
        # 
        z = self.z_proj(z)
        for layer in self.transformer_encoder:
            img_feats, state, z = layer(img_feats, state, z)
        x = torch.concat([img_feats, state[:, None], z[:, None]], 1)
        action_embeds = self.action_embeds.unsqueeze(0).expand(B, -1, -1)
        for layer in self.transformer_decoder:
            action_embeds = layer(action_embeds, context=x)
        return self.action_out_proj(action_embeds)  # (B, chunk_size, action_dim)

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return a single action for the current timestep (called at inference)."""
        with torch.no_grad():
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
        
        z_mean, log_sigma_x2_hat = self.cvae(state, actions, batch.get("action_is_pad"))
        if not use_mean:
            z = torch.distributions.Normal(z_mean, log_sigma_x2_hat.div(2).exp()).rsample()
        else:
            z = z_mean
        # x = torch.concat([img_feats, state[:, None], self.z_proj(z)[:, None]], 1)
        
        z = self.z_proj(z)
        for layer in self.transformer_encoder:
            img_feats, state, z = layer(img_feats, state, z)
        x = torch.concat([img_feats, state[:, None], z[:, None]], 1)
        
        # x = self.transformer_encoder(x)
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
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(512, 8, 1576),
            TransformerLayer(512, 8, 1576),
            TransformerLayer(512, 8, 1576),
            TransformerLayer(512, 8, 1576),
        ])
        self.final_proj = nn.Linear(512, 64)
        # self.register_buffer("cls_token", cls_token)

    def forward(self, state, actions, action_is_pad=None):
        B = state.shape[0]
        x = torch.concat([self.cls_token[None, None].expand(B, 1, -1), state[:, None], actions], 1)

        # Mask out action tokens that are padding (episode ended early) so they don't
        # corrupt the latent. The cls and state tokens (first two) are never padded.
        self_mask = None
        if action_is_pad is not None:
            cls_state_pad = torch.zeros(B, 2, dtype=torch.bool, device=action_is_pad.device)
            key_padding_mask = torch.cat([cls_state_pad, action_is_pad], dim=1)  # (B, 2 + chunk)
            self_mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, S), True = ignore

        for layer in self.transformer_layers:
            x = layer(x, self_mask=self_mask)
        z_mean, log_sigma_x2_hat = torch.split(self.final_proj(x[:, 0]), 32, 1)
        return z_mean, log_sigma_x2_hat
        
class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()
    
    def reset(self):
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.absolute_time = 0
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
            self.ensembled_actions[:, :-1] = self.ensembled_actions[:, 1:].clone()
            self.ensembled_actions[:, -1] = action_chunk[:, -1]
            self.ensembled_actions_count[:-1] = self.ensembled_actions_count[1:].clone()
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