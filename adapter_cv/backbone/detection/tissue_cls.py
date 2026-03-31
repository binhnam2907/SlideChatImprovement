"""TissueClassifier — MLP tissue-type classifier with
stratified sampling for diagnostically balanced patch selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TissueClassifier(nn.Module):
    """Predict tissue type per patch, then sample by quota."""

    TISSUE_TYPES = [
        'background', 'normal', 'stroma',
        'tumor', 'necrosis', 'lymphocyte',
    ]

    def __init__(self,
                 input_dim=512,
                 hidden_dim=256,
                 n_classes=6,
                 dropout=0.2,
                 sampling_weights=None):
        super().__init__()
        self.n_classes = n_classes

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        if sampling_weights is None:
            sampling_weights = [0.0, 0.1, 0.15, 1.0, 0.05, 0.3]
        self.register_buffer(
            'sampling_weights',
            torch.tensor(
                sampling_weights[:n_classes],
                dtype=torch.float32))

    def forward(self, features):
        """
        Returns:
            logits (B,N,C), probs (B,N,C), scores (B,N)
        """
        B, N, D = features.shape
        flat = features.reshape(B * N, D)
        logits = self.classifier(flat).reshape(
            B, N, self.n_classes)
        probs = F.softmax(logits, dim=-1)
        scores = (probs * self.sampling_weights).sum(dim=-1)
        return logits, probs, scores

    def stratified_select(self, features, total_k,
                          min_per_class=10):
        """Select patches with per-class quota sampling."""
        logits, probs, scores = self.forward(features)
        B, N, D = features.shape
        pred_classes = probs.argmax(dim=-1)

        all_selected = []
        all_indices = []

        for b in range(B):
            selected_idx = []
            remaining_k = min(total_k, N)

            for c in range(self.n_classes):
                w = self.sampling_weights[c].item()
                if w <= 0:
                    continue
                class_mask = (pred_classes[b] == c)
                class_idx = class_mask.nonzero(as_tuple=True)[0]
                if len(class_idx) == 0:
                    continue

                w_sum = self.sampling_weights.sum().item()
                quota = max(
                    int(remaining_k * w / w_sum),
                    min_per_class)
                quota = min(quota, len(class_idx))

                class_scores = probs[b, class_idx, c]
                _, top = torch.topk(class_scores, quota)
                selected_idx.append(class_idx[top])

            if selected_idx:
                sel = torch.cat(selected_idx)
                sel = sel.unique().sort().values
            else:
                _, sel = torch.topk(
                    scores[b], min(total_k, N))
                sel = sel.sort().values

            if len(sel) > total_k:
                perm = torch.randperm(
                    len(sel), device=sel.device)[:total_k]
                sel = sel[perm].sort().values

            all_selected.append(features[b, sel])
            all_indices.append(sel)

        max_len = max(s.shape[0] for s in all_selected)
        padded = []
        for s in all_selected:
            if s.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - s.shape[0], D,
                    device=s.device, dtype=s.dtype)
                s = torch.cat([s, pad])
            padded.append(s)

        selected = torch.stack(padded)
        return selected, scores, all_indices
