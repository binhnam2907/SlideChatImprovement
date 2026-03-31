import torch
import torch.nn as nn


class CVBackbone(nn.Module):
    """Full transformer encoder backbone.

    Applies multi-layer self-attention so every patch can attend
    to all other patches, producing context-enriched features.

    Best for: maximum representation quality when compute budget
    allows (Stage 1 alignment training).

    Architecture:
        LayerNorm -> TransformerEncoder x N -> LayerNorm
    """

    def __init__(self,
                 input_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 ff_dim: int = 2048,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim

        self.input_norm = nn.LayerNorm(input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True)

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)"""
        x = self.input_norm(x)
        x = self.encoder(x)
        x = self.output_norm(x)
        return x
