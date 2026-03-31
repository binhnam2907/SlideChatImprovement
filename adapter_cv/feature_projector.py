import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class FeatureProjector(nn.Module):
    """Projects CV backbone features into the LLM embedding space.

    Architecture: Linear(cv_dim → llm_dim) → GELU → Linear(llm_dim → llm_dim)

    This bridges the representational gap between the vision encoder's
    feature space and the language model's token embedding space so that
    visual tokens can be directly concatenated with text tokens.
    """

    def __init__(self,
                 cv_hidden_size: int = 512,
                 llm_hidden_size: int = 3584,
                 depth: int = 2,
                 hidden_act: str = 'gelu',
                 dropout: float = 0.1):
        super().__init__()
        self.cv_hidden_size = cv_hidden_size
        self.llm_hidden_size = llm_hidden_size

        layers = [nn.Linear(cv_hidden_size, llm_hidden_size)]
        for _ in range(1, depth):
            layers.append(ACT2FN[hidden_act])
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(llm_hidden_size, llm_hidden_size))

        self.proj = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, cv_hidden_size) — latent features from CVModel

        Returns:
            (B, N, llm_hidden_size) — features aligned with LLM embeddings
        """
        return self.proj(x)
