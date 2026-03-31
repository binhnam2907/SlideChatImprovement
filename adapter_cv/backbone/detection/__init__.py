"""Detection backbones — score and select relevant patches.

Input (B, N, D) -> selected (B, M, D), scores (B, N), indices.
"""

import torch.nn as nn

from .abmil import GatedAttentionSelector
from .clam import CLAMSelector
from .transmil import TransMILSelector
from .tissue_cls import TissueClassifier

DET_REGISTRY = {
    'abmil': GatedAttentionSelector,
    'clam': CLAMSelector,
    'transmil': TransMILSelector,
    'tissue_cls': TissueClassifier,
}


class PatchDetector(nn.Module):
    """Unified wrapper — single interface for all detectors."""

    def __init__(self,
                 method='abmil',
                 input_dim=512,
                 top_k=None,
                 top_ratio=0.7,
                 min_patches=128,
                 **kwargs):
        super().__init__()
        self.method = method
        self.top_k = top_k
        self.top_ratio = top_ratio

        if method not in DET_REGISTRY:
            raise ValueError(
                f"Unknown method: '{method}'. "
                f"Choose from: {list(DET_REGISTRY.keys())}")

        if method == 'tissue_cls':
            self.detector = TissueClassifier(
                input_dim=input_dim, **kwargs)
        else:
            self.detector = DET_REGISTRY[method](
                input_dim=input_dim,
                top_k=top_k,
                top_ratio=top_ratio,
                min_patches=min_patches,
                **kwargs)

    def forward(self, features, **kwargs):
        if self.method == 'tissue_cls':
            total_k = self.top_k or max(
                int(features.shape[1] * self.top_ratio), 128)
            return self.detector.stratified_select(
                features, total_k=total_k)

        if (self.method == 'clam'
                and kwargs.get('return_cluster_loss')):
            return self.detector(
                features, return_cluster_loss=True)

        return self.detector(features)

    def param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters()
            if p.requires_grad)
        return {'method': self.method,
                'total': total,
                'trainable': trainable}


def build_det_backbone(det_backbone='abmil',
                       input_dim=512,
                       top_k=None,
                       top_ratio=0.7,
                       **kw):
    """Build a detection backbone by name."""
    return PatchDetector(
        method=det_backbone,
        input_dim=input_dim,
        top_k=top_k,
        top_ratio=top_ratio,
        **kw)
