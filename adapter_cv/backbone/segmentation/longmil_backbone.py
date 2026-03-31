"""LongMIL-inspired backbone — local-global hybrid attention.

Reference:
    Qu et al., "Rethinking Transformer for Long Contextual
    Histopathology Whole Slide Image Analysis", NeurIPS 2024.

Alternates between local windowed attention (cheap, captures
neighbourhood) and global attention on downsampled landmarks
(captures slide-level context). Linear overall complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAttention(nn.Module):
    """Windowed self-attention over non-overlapping chunks."""

    def __init__(self, dim, num_heads=8, window_size=256,
                 dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        w = self.window_size

        pad = (w - N % w) % w
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        _, N_pad, _ = x.shape
        n_win = N_pad // w

        # (B, n_win, w, C)
        x = x.reshape(B, n_win, w, C)
        qkv = self.qkv(x).reshape(
            B, n_win, w, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(
            B, n_win, w, C).reshape(B, N_pad, C)

        if pad > 0:
            out = out[:, :N]
        return self.proj(out)


class GlobalPoolAttention(nn.Module):
    """Global attention via pooled summary tokens.

    Pools the sequence into ``num_globals`` summary tokens, applies
    cross-attention from full sequence to summaries, then broadcasts
    back. Cost: O(N * num_globals).
    """

    def __init__(self, dim, num_heads=8, num_globals=64,
                 dropout=0.1):
        super().__init__()
        self.num_globals = num_globals
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        g = min(self.num_globals, N)

        # Pool into g summary tokens via adaptive avg pool
        pooled = x.transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(pooled, g)
        pooled = pooled.transpose(1, 2)

        q = self.q_proj(x).reshape(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(pooled).reshape(
            B, g, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class LongMILBlock(nn.Module):
    """One local-then-global block with FFN."""

    def __init__(self, dim, num_heads=8, window_size=256,
                 num_globals=64, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.local_attn = LocalAttention(
            dim, num_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.global_attn = GlobalPoolAttention(
            dim, num_heads, num_globals, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.local_attn(self.norm1(x))
        x = x + self.global_attn(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x


class LongMILBackbone(nn.Module):
    """Local-global hybrid transformer (LongMIL-style).

    Each layer applies windowed local attention then cross-attends
    to a small set of pooled global summaries. Total complexity is
    O(N * (w + g)) which is linear in N for fixed window/global
    sizes.
    """

    def __init__(self,
                 input_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 ff_dim: int = 2048,
                 window_size: int = 256,
                 num_globals: int = 64,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim

        self.input_norm = nn.LayerNorm(input_dim)
        self.blocks = nn.ModuleList([
            LongMILBlock(input_dim, num_heads, window_size,
                         num_globals, ff_dim, dropout)
            for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)"""
        x = self.input_norm(x)
        for blk in self.blocks:
            x = blk(x)
        return self.output_norm(x)
