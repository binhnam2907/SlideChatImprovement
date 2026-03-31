"""Fusion strategies to merge segmentation and detection outputs.

The segmentation branch produces context-enriched features for
ALL patches: (B, N, D).

The detection branch produces relevance scores for ALL patches
and selects the top-M: (B, M, D) selected + (B, N) scores.

The fusion module combines both signals into a single output
that is passed to the FeatureProjector.

Available strategies:
    det_guided   Use detection indices to pick from segmented
                 features. Output is (B, M, D). Best of both
                 worlds: context from seg, selection from det.

    score_weight Multiply segmentation features by detection
                 scores (soft attention). Output is (B, N, D).
                 Keeps all patches but emphasises relevant ones.

    concat       Concatenate seg-selected and det-selected
                 features, project back to D. Output is (B, M, D).

    gated        Learned gate blends seg and det per patch.
                 Output is (B, M, D).

    add          Element-wise add of seg-selected and det-selected
                 features (simplest). Output is (B, M, D).
"""

import torch
import torch.nn as nn


class DetGuidedFusion(nn.Module):
    """Use detection indices to select from segmented features.

    seg_output: (B, N, D) — all patches, context-enriched
    det_indices: list[Tensor] — which M patches detection chose

    Result: (B, M, D) — the M selected patches but with
    segmentation's richer representations.
    """

    def __init__(self, input_dim=512, **kwargs):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, seg_out, det_out, det_scores, det_indices):
        B, N, D = seg_out.shape
        selected = []
        for b in range(B):
            idx = det_indices[b]
            selected.append(seg_out[b, idx])
        return torch.stack(selected)


class ScoreWeightFusion(nn.Module):
    """Soft-weight all segmentation features by detection scores.

    Does NOT reduce the sequence length — keeps all N patches
    but rescales each by its detection relevance. Useful when
    the LLM context window can handle all N patches.
    """

    def __init__(self, input_dim=512, temperature=1.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.temperature = temperature

    def forward(self, seg_out, det_out, det_scores, det_indices):
        weights = (det_scores / self.temperature).softmax(dim=-1)
        weights = weights.unsqueeze(-1)
        return seg_out * weights


class ConcatFusion(nn.Module):
    """Concatenate seg-selected + det-selected, then project.

    Takes the M patches selected by detection from BOTH branches,
    concatenates their feature vectors [seg; det], and projects
    back to D. This gives the model two views of each patch.
    """

    def __init__(self, input_dim=512, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
        )

    def forward(self, seg_out, det_out, det_scores, det_indices):
        B = seg_out.shape[0]
        seg_sel = []
        for b in range(B):
            idx = det_indices[b]
            seg_sel.append(seg_out[b, idx])
        seg_sel = torch.stack(seg_sel)

        merged = torch.cat([seg_sel, det_out], dim=-1)
        return self.proj(merged)


class GatedFusion(nn.Module):
    """Learned gate blends segmentation and detection features.

    For each of the M selected patches, a sigmoid gate decides
    how much to take from the segmentation branch vs. the
    detection branch.

    gate = sigmoid(W_s * seg + W_d * det + bias)
    output = gate * seg + (1 - gate) * det
    """

    def __init__(self, input_dim=512, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.gate_seg = nn.Linear(input_dim, input_dim)
        self.gate_det = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, seg_out, det_out, det_scores, det_indices):
        B = seg_out.shape[0]
        seg_sel = []
        for b in range(B):
            idx = det_indices[b]
            seg_sel.append(seg_out[b, idx])
        seg_sel = torch.stack(seg_sel)

        gate = torch.sigmoid(
            self.gate_seg(seg_sel) + self.gate_det(det_out))
        fused = gate * seg_sel + (1.0 - gate) * det_out
        return self.norm(fused)


class AddFusion(nn.Module):
    """Simple element-wise addition of both branches.

    Takes the M patches selected by detection from the
    segmentation output, and adds the detection output.
    """

    def __init__(self, input_dim=512, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, seg_out, det_out, det_scores, det_indices):
        B = seg_out.shape[0]
        seg_sel = []
        for b in range(B):
            idx = det_indices[b]
            seg_sel.append(seg_out[b, idx])
        seg_sel = torch.stack(seg_sel)
        return self.norm(seg_sel + det_out)


FUSION_REGISTRY = {
    'det_guided': DetGuidedFusion,
    'score_weight': ScoreWeightFusion,
    'concat': ConcatFusion,
    'gated': GatedFusion,
    'add': AddFusion,
}


def build_fusion(fusion_mode: str = 'det_guided',
                 input_dim: int = 512, **kwargs):
    """Build a fusion module by name."""
    if fusion_mode not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion_mode='{fusion_mode}'. "
            f"Choose from: {list(FUSION_REGISTRY.keys())}")
    return FUSION_REGISTRY[fusion_mode](
        input_dim=input_dim, **kwargs)
