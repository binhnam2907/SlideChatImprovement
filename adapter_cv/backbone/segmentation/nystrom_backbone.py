"""Nystrom-attention backbone — O(N) approximate self-attention.

References:
    Xiong et al., "Nystromformer: A Nystrom-Based Algorithm for
    Approximating Self-Attention", AAAI 2021.

    Tan et al., "FALFormer: Feature-aware Landmarks Self-Attention
    for Whole-slide Image Classification", 2024.

Approximates the full N x N attention matrix by sampling a small
set of landmark tokens and reconstructing attention through the
Nystrom method.  This gives O(N * m) cost where m << N.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NystromAttention(nn.Module):
    """Single-head Nystrom self-attention."""

    def __init__(self, dim, num_heads=8, num_landmarks=64,
                 dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_landmarks = num_landmarks
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        m = min(self.num_landmarks, N)

        if m >= N:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v
        else:
            # Nystrom approximation
            idx = torch.linspace(
                0, N - 1, m, device=x.device).long()
            q_land = q[:, :, idx]
            k_land = k[:, :, idx]

            kernel_1 = F.softmax(
                (q @ k_land.transpose(-2, -1)) * self.scale,
                dim=-1)
            kernel_2 = F.softmax(
                (q_land @ k_land.transpose(-2, -1)) * self.scale,
                dim=-1)
            kernel_3 = F.softmax(
                (q_land @ k.transpose(-2, -1)) * self.scale,
                dim=-1)

            # pseudo-inverse of kernel_2
            k2_inv = torch.linalg.pinv(kernel_2)
            out = kernel_1 @ k2_inv @ kernel_3 @ v
            out = self.attn_drop(out)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class NystromBlock(nn.Module):
    """Pre-norm block with Nystrom attention + FFN."""

    def __init__(self, dim, num_heads=8, num_landmarks=64,
                 ff_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim, num_heads, num_landmarks, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class NystromBackbone(nn.Module):
    """Stacked Nystrom-attention encoder.

    Provides the same contextual modelling as the full transformer
    backbone but at O(N * m) cost, making it practical for very
    large patch sets (> 20 K patches).
    """

    def __init__(self,
                 input_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 ff_dim: int = 2048,
                 num_landmarks: int = 64,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim

        self.input_norm = nn.LayerNorm(input_dim)
        self.blocks = nn.ModuleList([
            NystromBlock(input_dim, num_heads, num_landmarks,
                         ff_dim, dropout)
            for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)"""
        x = self.input_norm(x)
        for blk in self.blocks:
            x = blk(x)
        return self.output_norm(x)
