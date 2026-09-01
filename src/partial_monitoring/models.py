from __future__ import annotations

import torch
from torch import nn

from .policy import ACTION_NAMES


class NeuralRewardModel(nn.Module):
    """Reward model used by NeuralUCB on top of frozen BERT embeddings."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ACTION_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
