"""Default config for training_adapter.py.

Usage:
    python training_adapter.py \
        --data_path <path_to_train.json> \
        --seg_backbone transformer \
        --det_backbone abmil \
        --fusion_mode gated
"""

cv_model = dict(
    seg_backbone='transformer',
    det_backbone='abmil',
    fusion_mode='gated',
    cv_input_dim=512,
    cv_num_heads=8,
    cv_num_layers=4,
    cv_ff_dim=2048,
    cv_dropout=0.1,
    llm_hidden_size=512,
    proj_depth=1,
    use_residual=True,
    det_top_ratio=0.7,
    det_top_k=None,
)
