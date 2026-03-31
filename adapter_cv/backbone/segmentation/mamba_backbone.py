"""MambaMIL-inspired backbone — linear-complexity sequence modelling.

Reference:
    Yang et al., "MambaMIL: Enhancing Long Sequence Modeling with
    Sequence Reordering in Computational Pathology", MICCAI 2024.

Uses a simplified selective state-space layer (S6-style) that
scans the patch sequence in both forward and backward directions,
capturing long-range dependencies with O(N) cost instead of the
O(N^2) of self-attention.

When the full ``mamba_ssm`` package is not installed the module
falls back to a GRU-based scan that preserves the same API and
similar inductive bias (sequential gating over hidden state).
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class _GRUScan(nn.Module):
    """Fallback bidirectional GRU scan when mamba_ssm is absent."""

    def __init__(self, d_model, d_state=16, **kwargs):
        super().__init__()
        self.gru = nn.GRU(
            d_model, d_model, batch_first=True,
            bidirectional=True)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.proj(h)


class _MambaBlock(nn.Module):
    """Single Mamba block: norm -> SSM -> residual."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if MAMBA_AVAILABLE:
            self.ssm = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand)
        else:
            self.ssm = _GRUScan(d_model, d_state)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class MambaBackbone(nn.Module):
    """Stacked Mamba blocks with bidirectional scanning.

    Processes the patch sequence in both forward and reverse order,
    then averages the two directions — analogous to the
    Sequence Reordering idea from MambaMIL.
    """

    def __init__(self,
                 input_dim: int = 512,
                 num_layers: int = 4,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim

        self.fwd_blocks = nn.ModuleList([
            _MambaBlock(input_dim, d_state, d_conv, expand)
            for _ in range(num_layers)])
        self.bwd_blocks = nn.ModuleList([
            _MambaBlock(input_dim, d_state, d_conv, expand)
            for _ in range(num_layers)])

        self.merge = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(dropout))
        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)"""
        fwd = x
        for blk in self.fwd_blocks:
            fwd = blk(fwd)

        bwd = x.flip(1)
        for blk in self.bwd_blocks:
            bwd = blk(bwd)
        bwd = bwd.flip(1)

        merged = self.merge(torch.cat([fwd, bwd], dim=-1))
        return self.output_norm(merged)
