"""Untrained continuation-value interface for future depth-limited CFR."""

import torch
from torch import nn


class ValueNetwork(nn.Module):
    """Maps an encoded public-belief state to one value per player.

    This architecture is intentionally small and is not trained or used by the
    current exact Kuhn traversal.
    """

    def __init__(self, input_size: int, hidden_size: int = 128, players: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, players),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
