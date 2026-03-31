import os
import torch
import torch.nn as nn

from .backbone import build_seg_backbone, build_det_backbone
from .fusion import build_fusion
from .feature_projector import FeatureProjector


class CVModel(nn.Module):
    """Dual-branch vision adapter: segmentation + detection + fusion.

    Two parallel branches process the raw patch features:

        Segmentation branch  — refines ALL patches with context
        Detection branch     — scores patches, selects top-M

    A fusion module merges both outputs before projection to the
    LLM embedding space.

    Pipeline:
        raw features (B, N, 512)
            ├─ seg_backbone  → seg_out (B, N, 512)   [all patches]
            ├─ det_backbone  → det_out (B, M, 512)   [selected]
            │                  + scores (B, N)
            │                  + indices
            └─ fusion(seg_out, det_out, scores, indices)
                            → fused (B, M, 512)
            → FeatureProjector → (B, M, 3584)

    When only one branch is enabled, the model degrades gracefully:
    - seg only:  seg → project  (no selection, all patches kept)
    - det only:  det → project  (no context enrichment)
    - neither:   raw → project  (passthrough)
    """

    def __init__(self,
                 seg_backbone: str = 'transformer',
                 det_backbone: str = None,
                 fusion_mode: str = 'det_guided',
                 cv_input_dim: int = 512,
                 cv_num_heads: int = 8,
                 cv_num_layers: int = 4,
                 cv_ff_dim: int = 2048,
                 cv_dropout: float = 0.1,
                 llm_hidden_size: int = 3584,
                 proj_depth: int = 2,
                 proj_dropout: float = 0.1,
                 use_residual: bool = True,
                 det_top_k: int = None,
                 det_top_ratio: float = 0.7,
                 det_kwargs: dict = None,
                 seg_kwargs: dict = None,
                 fusion_kwargs: dict = None,
                 # keep old API aliases working
                 backbone_type: str = None,
                 detector_method: str = None,
                 detector_top_k: int = None,
                 detector_top_ratio: float = 0.7,
                 detector_kwargs: dict = None,
                 backbone_kwargs: dict = None):
        super().__init__()

        # --- resolve old API aliases ---
        if backbone_type is not None and seg_backbone == 'transformer':
            seg_backbone = backbone_type
        if detector_method is not None and det_backbone is None:
            det_backbone = detector_method
        if detector_top_k is not None and det_top_k is None:
            det_top_k = detector_top_k
        if detector_top_ratio != 0.7 and det_top_ratio == 0.7:
            det_top_ratio = detector_top_ratio
        if detector_kwargs and det_kwargs is None:
            det_kwargs = detector_kwargs
        if backbone_kwargs and seg_kwargs is None:
            seg_kwargs = backbone_kwargs

        self.seg_backbone_name = seg_backbone
        self.det_backbone_name = det_backbone
        self.fusion_mode = fusion_mode
        self.use_residual = use_residual
        self.cv_input_dim = cv_input_dim

        self._init_config = dict(
            seg_backbone=seg_backbone,
            det_backbone=det_backbone,
            fusion_mode=fusion_mode,
            cv_input_dim=cv_input_dim,
            cv_num_heads=cv_num_heads,
            cv_num_layers=cv_num_layers,
            cv_ff_dim=cv_ff_dim,
            cv_dropout=cv_dropout,
            llm_hidden_size=llm_hidden_size,
            proj_depth=proj_depth,
            proj_dropout=proj_dropout,
            use_residual=use_residual,
            det_top_k=det_top_k,
            det_top_ratio=det_top_ratio,
        )

        # --- segmentation branch (optional) ---
        self.seg = None
        if seg_backbone is not None:
            seg_kw = dict(
                input_dim=cv_input_dim,
                num_heads=cv_num_heads,
                num_layers=cv_num_layers,
                ff_dim=cv_ff_dim,
                dropout=cv_dropout,
            )
            if seg_kwargs:
                seg_kw.update(seg_kwargs)
            self.seg = build_seg_backbone(seg_backbone, **seg_kw)

        # --- detection branch (optional) ---
        self.det = None
        if det_backbone is not None:
            det_kw = det_kwargs or {}
            self.det = build_det_backbone(
                det_backbone,
                input_dim=cv_input_dim,
                top_k=det_top_k,
                top_ratio=det_top_ratio,
                **det_kw)

        # --- fusion (only when both branches exist) ---
        self.fusion = None
        if self.seg is not None and self.det is not None:
            fus_kw = fusion_kwargs or {}
            self.fusion = build_fusion(
                fusion_mode,
                input_dim=cv_input_dim,
                **fus_kw)

        # --- projector ---
        self.projector = FeatureProjector(
            cv_hidden_size=cv_input_dim,
            llm_hidden_size=llm_hidden_size,
            depth=proj_depth,
            dropout=proj_dropout)

    def forward(self, patch_features, return_extras=False):
        """
        Args:
            patch_features: (B, N, cv_input_dim)
            return_extras: return intermediate outputs

        Returns:
            projected: (B, M, llm_hidden_size)
            extras (optional): dict with seg_out, det_out,
                det_scores, det_indices, fused
        """
        extras = {}
        seg_out = None
        det_out = None
        det_scores = None
        det_indices = None

        # --- segmentation branch ---
        if self.seg is not None:
            seg_out = self.seg(patch_features)
            if self.use_residual:
                seg_out = seg_out + patch_features
            extras['seg_out'] = seg_out

        # --- detection branch ---
        if self.det is not None:
            det_out, det_scores, det_indices = self.det(
                patch_features)
            extras['det_out'] = det_out
            extras['det_scores'] = det_scores
            extras['det_indices'] = det_indices

        # --- merge ---
        if self.fusion is not None:
            fused = self.fusion(
                seg_out, det_out, det_scores, det_indices)
            extras['fused'] = fused
        elif seg_out is not None and det_out is not None:
            fused = det_out
        elif seg_out is not None:
            fused = seg_out
        elif det_out is not None:
            fused = det_out
        else:
            fused = patch_features

        projected = self.projector(fused)

        if return_extras:
            return projected, extras
        return projected

    def param_count(self) -> dict:
        seg_p = (sum(p.numel() for p in self.seg.parameters())
                 if self.seg else 0)
        det_p = (sum(p.numel() for p in self.det.parameters())
                 if self.det else 0)
        fus_p = (sum(p.numel() for p in self.fusion.parameters())
                 if self.fusion else 0)
        proj_p = sum(p.numel() for p in self.projector.parameters())
        total = seg_p + det_p + fus_p + proj_p
        trainable = sum(
            p.numel() for p in self.parameters()
            if p.requires_grad)
        return {
            'segmentation': seg_p,
            'detection': det_p,
            'fusion': fus_p,
            'projector': proj_p,
            'total': total,
            'trainable': trainable,
        }

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.state_dict(),
            os.path.join(save_dir, 'cv_model.pt'))
        torch.save(
            self._init_config,
            os.path.join(save_dir, 'cv_config.pt'))

    @classmethod
    def load(cls, save_dir: str, device='cpu'):
        config = torch.load(
            os.path.join(save_dir, 'cv_config.pt'),
            map_location=device, weights_only=True)
        model = cls(**config)
        state = torch.load(
            os.path.join(save_dir, 'cv_model.pt'),
            map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model
