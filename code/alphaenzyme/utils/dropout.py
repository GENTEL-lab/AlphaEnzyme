# Adapted from OpenFold's dropout module (Apache-2.0).

import torch
import torch.nn as nn
from typing import List, Union


class Dropout(nn.Module):
    """Dropout with an optional shared mask along selected dimensions."""

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        super().__init__()
        self.r = r
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x = x * mask
        return x


class DropoutRowwise(Dropout):
    def __init__(self, r: float):
        super().__init__(r, batch_dim=-3)


class DropoutColumnwise(Dropout):
    def __init__(self, r: float):
        super().__init__(r, batch_dim=-2)
