import torch
import torch.nn as nn


class LightweightBackbone(nn.Module):
    """Two-layer MLP backbone with no attention.

    Applies independent per-patch transformations (no cross-patch
    interaction). Much faster and lighter than the transformer
    variant — useful as a quick baseline or when the detector
    already provides enough context.

    Architecture:
        LayerNorm -> Linear -> GELU -> Dropout
                  -> Linear -> LayerNorm
    """

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 1024,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)"""
        return self.net(x)
