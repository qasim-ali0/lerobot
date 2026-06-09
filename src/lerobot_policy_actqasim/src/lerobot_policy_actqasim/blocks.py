import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, d, max_seq_len=512):
        super().__init__()
        self.d = d
        indices = torch.arange(0, d // 2).float()
        omegas = torch.exp(-2 * indices / self.d * torch.log(torch.tensor(300.0)))
        positions = torch.arange(0, max_seq_len).float()
        angles = torch.outer(positions, omegas)
        self.register_buffer('cos_angles', angles.cos())
        self.register_buffer('sin_angles', angles.sin())

    def forward(self, x):
        # x: (B, T, D)
        seq_len = x.shape[1]
        cos = self.cos_angles[:seq_len]  # (T, D//2)
        sin = self.sin_angles[:seq_len]
        odds = x[:, :, ::2].clone()
        evens = x[:, :, 1::2].clone()
        x_rot = torch.stack([
            odds * cos - evens * sin,
            evens * cos + odds * sin,
        ], dim=-1).flatten(-2)
        return x_rot


class Attention(nn.Module):
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
        self.rope = RoPE(self.d_head)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, n_heads, T, d_head)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, T, D) query input
            context: (B, S, D) key/value source — if None, self-attention
            mask:    (B, 1, T, S) or broadcastable boolean mask (True = ignore)
        Returns:
            (B, T, D)
        """
        kv_src = x if context is None else context
        B, T, _ = x.shape

        q = self._split_heads(self.q(x))        # (B, H, T, d_head)
        k = self._split_heads(self.k(kv_src))   # (B, H, S, d_head)
        v = self._split_heads(self.v(kv_src))   # (B, H, S, d_head)

        # Apply RoPE to q and k along the head dimension
        q = self.rope(q.flatten(0, 1)).view(B, self.n_heads, -1, self.d_head)
        k = self.rope(k.flatten(0, 1)).view(B, self.n_heads, -1, self.d_head)

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, H, T, S)

        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = attn @ v                            # (B, H, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer: self-attention + optional cross-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_attn = Attention(d_model, n_heads, dropout)
            self.norm_cross = nn.LayerNorm(d_model)

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
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention (pre-norm)
        x = x + self.self_attn(self.norm1(x), mask=self_mask)

        # Cross-attention (pre-norm), only if enabled and context provided
        if self.cross_attention and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context=context, mask=cross_mask)

        # FFN (pre-norm)
        x = x + self.ff(self.norm2(x))
        return x
