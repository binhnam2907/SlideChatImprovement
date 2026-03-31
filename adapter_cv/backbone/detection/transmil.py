"""TransMIL — Transformer-Based MIL Selector.

Reference:
    Shao et al., "TransMIL: Transformer based Correlated
    Multiple Instance Learning for Whole Slide Image
    Classification", NeurIPS 2021.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator.

    Injects local spatial priors via depth-wise convolutions at
    multiple kernel sizes (3, 5, 7).
    """

    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        cls_token = x[:, :1]
        feat = x[:, 1:]

        H = W = int(math.ceil(math.sqrt(feat.shape[1])))
        pad_len = H * W - feat.shape[1]
        if pad_len > 0:
            feat = F.pad(feat, (0, 0, 0, pad_len))

        feat = feat.transpose(1, 2).view(B, C, H, W)
        feat = (self.proj(feat) + feat
                + self.proj1(feat) + self.proj2(feat))
        feat = feat.flatten(2).transpose(1, 2)

        if pad_len > 0:
            feat = feat[:, :N - 1]

        return torch.cat([cls_token, feat], dim=1)


class TransMILSelector(nn.Module):
    """Transformer + PPEG + CLS-token attention selector."""

    def __init__(self,
                 input_dim=512,
                 hidden_dim=512,
                 num_heads=8,
                 num_layers=2,
                 dropout=0.1,
                 top_k=None,
                 top_ratio=0.7,
                 min_patches=128):
        super().__init__()
        self.top_k = top_k
        self.top_ratio = top_ratio
        self.min_patches = min_patches

        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim))
        self.ppeg = PPEG(dim=hidden_dim)

        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True,
            norm_first=True)

        self.layer1 = nn.TransformerEncoder(enc, num_layers=1)
        self.layer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True),
            num_layers=max(num_layers - 1, 1))

        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def _compute_k(self, N):
        if self.top_k is not None:
            return min(self.top_k, N)
        return min(
            max(int(N * self.top_ratio), self.min_patches), N)

    def forward(self, features):
        """
        Returns:
            selected (B, M, D), attn (B, N), indices list[Tensor]
        """
        B, N, D = features.shape
        h = self.fc_in(features)

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)

        h = self.layer1(h)
        h = self.ppeg(h)
        h = self.layer2(h)
        h = self.norm(h)

        patch_h = h[:, 1:]
        scores = self.attn_score(patch_h).squeeze(-1)
        attn = F.softmax(scores, dim=-1)

        k = self._compute_k(N)
        all_selected = []
        all_indices = []
        for b in range(B):
            _, sel_idx = torch.topk(attn[b], k)
            sel_idx = sel_idx.sort().values
            all_selected.append(features[b, sel_idx])
            all_indices.append(sel_idx)

        selected = torch.stack(all_selected)
        return selected, attn, all_indices
