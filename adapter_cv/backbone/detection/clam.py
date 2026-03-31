"""CLAM — Clustering-constrained Attention MIL Selector.

Reference:
    Lu et al., "Data-efficient and weakly supervised computational
    pathology on whole-slide images",
    Nature Biomedical Engineering, 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLAMSelector(nn.Module):
    """Multi-class attention + instance-level clustering loss."""

    def __init__(self,
                 input_dim=512,
                 hidden_dim=256,
                 dropout=0.25,
                 n_classes=2,
                 subtyping=False,
                 top_k=None,
                 top_ratio=0.7,
                 min_patches=128,
                 inst_cluster_k=8):
        super().__init__()
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.inst_cluster_k = inst_cluster_k
        self.top_k = top_k
        self.top_ratio = top_ratio
        self.min_patches = min_patches

        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention_branches = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_classes)
        ])

        self.classifiers = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(n_classes)
        ])

        self.instance_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            ) for _ in range(n_classes)
        ])

    def _compute_k(self, N):
        if self.top_k is not None:
            return min(self.top_k, N)
        return min(
            max(int(N * self.top_ratio), self.min_patches), N)

    def compute_attention(self, features):
        B, N, D = features.shape
        h = self.attention_net(features)

        class_attns = []
        for branch in self.attention_branches:
            a = branch(h).squeeze(-1)
            a = F.softmax(a, dim=-1)
            class_attns.append(a)

        attn_all = torch.stack(class_attns, dim=1)
        combined = attn_all.mean(dim=1)
        return attn_all, combined

    def instance_clustering_loss(self, features, attn_weights):
        B, N, D = features.shape
        total_loss = torch.tensor(0.0, device=features.device)
        k = min(self.inst_cluster_k, N // 2)
        if k == 0:
            return total_loss

        for b in range(B):
            for c in range(self.n_classes):
                a = attn_weights[b, c]
                _, top_idx = torch.topk(a, k)
                _, bot_idx = torch.topk(a, k, largest=False)

                inst_feats = torch.cat(
                    [features[b, top_idx], features[b, bot_idx]])
                inst_labels = torch.cat([
                    torch.ones(k, device=features.device),
                    torch.zeros(k, device=features.device),
                ]).long()

                logits = self.instance_classifiers[c](inst_feats)
                loss = F.cross_entropy(logits, inst_labels)
                total_loss = total_loss + loss

        return total_loss / (B * self.n_classes)

    def forward(self, features, return_cluster_loss=False):
        """
        Returns:
            selected (B, M, D), combined_attn (B, N),
            indices list[Tensor], [cluster_loss]
        """
        attn_all, combined = self.compute_attention(features)
        B, N, D = features.shape
        k = self._compute_k(N)

        all_selected = []
        all_indices = []
        for b in range(B):
            _, sel_idx = torch.topk(combined[b], k)
            sel_idx = sel_idx.sort().values
            all_selected.append(features[b, sel_idx])
            all_indices.append(sel_idx)

        selected = torch.stack(all_selected)

        if return_cluster_loss:
            cl_loss = self.instance_clustering_loss(
                features, attn_all)
            return selected, combined, all_indices, cl_loss

        return selected, combined, all_indices
