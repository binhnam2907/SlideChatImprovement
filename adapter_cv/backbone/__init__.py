"""Backbone package — segmentation and detection under one roof.

Usage:
    from adapter_cv.backbone import build_seg_backbone
    from adapter_cv.backbone import build_det_backbone

    seg = build_seg_backbone('transformer', input_dim=512)
    det = build_det_backbone('abmil', input_dim=512)
"""

from .segmentation import (
    CVBackbone,
    LightweightBackbone,
    ConvBackbone,
    MambaBackbone,
    NystromBackbone,
    LongMILBackbone,
    GraphBackbone,
    SEG_REGISTRY,
    build_seg_backbone,
)

from .detection import (
    GatedAttentionSelector,
    CLAMSelector,
    TransMILSelector,
    TissueClassifier,
    PatchDetector,
    DET_REGISTRY,
    build_det_backbone,
)

__all__ = [
    # segmentation
    'CVBackbone', 'LightweightBackbone', 'ConvBackbone',
    'MambaBackbone', 'NystromBackbone', 'LongMILBackbone',
    'GraphBackbone', 'SEG_REGISTRY', 'build_seg_backbone',
    # detection
    'GatedAttentionSelector', 'CLAMSelector',
    'TransMILSelector', 'TissueClassifier',
    'PatchDetector', 'DET_REGISTRY', 'build_det_backbone',
]
