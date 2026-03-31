"""Segmentation backbones — refine ALL patches with context.

Input (B, N, D) -> Output (B, N, D), same shape.
"""

from .cv_backbone import CVBackbone
from .lightweight_backbone import LightweightBackbone
from .conv_backbone import ConvBackbone
from .mamba_backbone import MambaBackbone
from .nystrom_backbone import NystromBackbone
from .longmil_backbone import LongMILBackbone
from .graph_backbone import GraphBackbone

SEG_REGISTRY = {
    'transformer': CVBackbone,
    'lightweight': LightweightBackbone,
    'conv': ConvBackbone,
    'mamba': MambaBackbone,
    'nystrom': NystromBackbone,
    'longmil': LongMILBackbone,
    'graph': GraphBackbone,
}


def build_seg_backbone(seg_backbone='transformer', **kw):
    """Build a segmentation backbone by name."""
    if seg_backbone not in SEG_REGISTRY:
        raise ValueError(
            f"Unknown seg_backbone='{seg_backbone}'. "
            f"Choose from: {list(SEG_REGISTRY.keys())}")
    return SEG_REGISTRY[seg_backbone](**kw)
