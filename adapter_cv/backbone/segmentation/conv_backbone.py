import torch
import torch.nn as nn


class ConvBackbone(nn.Module):
    """1-D convolutional backbone with local context.

    Reshapes the patch sequence into a 1-D signal and applies
    depth-wise separable convolutions at multiple kernel sizes
    to capture local neighbourhood patterns. Cheaper than full
    self-attention while still providing cross-patch context
    within a local window.

    Best for: large patch counts where transformer attention
    is too expensive but local context still matters.

    Architecture:
        LayerNorm -> Multi-scale DepthwiseConv1d -> PointwiseConv
        -> GELU -> Dropout -> PointwiseConv -> LayerNorm
    """

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 1024,
                 kernel_sizes: tuple = (3, 5, 7),
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim

        self.input_norm = nn.LayerNorm(input_dim)

        self.dw_convs = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim,
                      kernel_size=k,
                      padding=k // 2,
                      groups=input_dim)
            for k in kernel_sizes
        ])

        n_branches = len(kernel_sizes)
        self.pw_in = nn.Conv1d(
            input_dim * n_branches, hidden_dim, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.pw_out = nn.Conv1d(hidden_dim, input_dim, 1)

        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)"""
        x = self.input_norm(x)

        # (B, N, D) -> (B, D, N) for Conv1d
        h = x.transpose(1, 2)

        branches = [conv(h) for conv in self.dw_convs]
        h = torch.cat(branches, dim=1)

        h = self.pw_in(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.pw_out(h)

        # (B, D, N) -> (B, N, D)
        h = h.transpose(1, 2)
        h = self.output_norm(h)
        return h
