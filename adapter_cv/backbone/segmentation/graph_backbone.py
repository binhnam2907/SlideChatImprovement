"""PatchGCN-inspired graph backbone — spatial-aware message passing.

Reference:
    Chen et al., "Whole Slide Images are 2D Point Clouds:
    Context-Aware Survival Prediction using Patch-based Graph
    Convolutional Networks", MICCAI 2021.

Builds a k-NN graph over patch features (treating them as a
point cloud) and applies message-passing layers so each patch
aggregates information from its nearest neighbours. Works without
explicit spatial coordinates — neighbours are found in feature
space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Single graph convolution layer (GCN-style).

    For each node, aggregates neighbour features via mean pooling,
    concatenates with the node's own features, and projects.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Args:
            x: (B, N, D) node features
            adj: (B, N, K) neighbour indices
        Returns:
            (B, N, D) updated features
        """
        B, N, D = x.shape
        K = adj.shape[-1]

        # Gather neighbour features
        idx = adj.unsqueeze(-1).expand(B, N, K, D)
        neigh = torch.gather(
            x.unsqueeze(2).expand(B, N, N, D),
            dim=2,
            index=idx)
        neigh_mean = neigh.mean(dim=2)

        out = torch.cat([x, neigh_mean], dim=-1)
        out = self.linear(out)
        out = F.gelu(out)
        out = self.drop(out)
        return self.norm(out + x)


class GraphBackbone(nn.Module):
    """Stacked graph convolution layers with k-NN adjacency.

    Builds adjacency in feature space (not spatial coordinates)
    so it works on the pre-computed CONCH embeddings without
    needing patch (x, y) positions.
    """

    def __init__(self,
                 input_dim: int = 512,
                 num_layers: int = 4,
                 k_neighbours: int = 8,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.k = k_neighbours

        self.input_norm = nn.LayerNorm(input_dim)
        self.layers = nn.ModuleList([
            GraphConvLayer(input_dim, dropout)
            for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(input_dim)

    def _build_knn(self, x):
        """Build k-NN adjacency from cosine similarity.

        Returns:
            adj: (B, N, K) indices of K nearest neighbours.
        """
        x_norm = F.normalize(x, dim=-1)
        sim = x_norm @ x_norm.transpose(-2, -1)
        sim.diagonal(dim1=-2, dim2=-1).fill_(-1e9)
        k = min(self.k, x.shape[1] - 1)
        _, adj = sim.topk(k, dim=-1)
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)"""
        x = self.input_norm(x)
        adj = self._build_knn(x)
        for layer in self.layers:
            x = layer(x, adj)
        return self.output_norm(x)
