"""Dataset contract for future solver-generated continuation values."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ValueExample:
    public_belief_state: Sequence[float]
    values: Sequence[float]


class ValueDataset(Dataset):
    def __init__(self, examples: Sequence[ValueExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.examples[index]
        return (torch.tensor(item.public_belief_state, dtype=torch.float32),
                torch.tensor(item.values, dtype=torch.float32))
