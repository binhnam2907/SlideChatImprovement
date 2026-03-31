"""ABMIL — Gated Attention-Based MIL Selector.

Reference:
    Ilse et al., "Attention-based Deep Multiple Instance
    Learning", ICML 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    """Two-branch gated attention (tanh × sigmoid)."""

    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """(B, N, D) -> attention (B, N), raw_scores (B, N)"""
        a_V = self.attention_V(x)
        a_U = self.attention_U(x)
        raw = self.attention_w(a_V * a_U).squeeze(-1)
        attention = F.softmax(raw, dim=-1)
        return attention, raw


class GatedAttentionSelector(nn.Module):
    """Score every patch via gated attention, select top-K + random."""

    def __init__(self,
                 input_dim=512,
                 hidden_dim=256,
                 dropout=0.25,
                 top_k=None,
                 top_ratio=0.7,
                 min_patches=128,
                 random_ratio=0.1):
        super().__init__()
        self.attention = GatedAttention(input_dim, hidden_dim, dropout)
        self.top_k = top_k
        self.top_ratio = top_ratio
        self.min_patches = min_patches
        self.random_ratio = random_ratio

    def _compute_k(self, N):
        if self.top_k is not None:
            return min(self.top_k, N)
        return min(max(int(N * self.top_ratio), self.min_patches), N)

    def forward(self, features):
        """
        Returns:
            selected (B, M, D), attn (B, N), indices list[Tensor]
        """
        attn_weights, _ = self.attention(features)
        B, N, D = features.shape
        k = self._compute_k(N)
        n_rand = max(int(k * self.random_ratio), 1)
        n_top = k - n_rand

        all_selected = []
        all_indices = []

        for b in range(B):
            _, top_idx = torch.topk(attn_weights[b], n_top)

            mask = torch.ones(N, dtype=torch.bool,
                              device=features.device)
            mask[top_idx] = False
            remaining = mask.nonzero(as_tuple=True)[0]

            if len(remaining) >= n_rand:
                perm = torch.randperm(
                    len(remaining),
                    device=features.device)[:n_rand]
                rand_idx = remaining[perm]
            else:
                rand_idx = remaining

            sel_idx = torch.cat([top_idx, rand_idx]).sort().values
            all_selected.append(features[b, sel_idx])
            all_indices.append(sel_idx)

        selected = torch.stack(all_selected)
        return selected, attn_weights, all_indices
