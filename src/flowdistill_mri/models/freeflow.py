from __future__ import annotations

import torch
from torch import nn

from flowdistill_mri.flows.interpolation import append_dims


class PriorAnchoredFreeFlow(nn.Module):
    """Paper-faithful Track A parameterization anchored at a prior sample."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def average_velocity(
        self, noise: torch.Tensor, delta: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.backbone(noise, delta, condition)

    def map_from_prior(
        self, noise: torch.Tensor, delta: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        return noise + append_dims(delta, noise) * self.average_velocity(noise, delta, condition)

    def forward(
        self, noise: torch.Tensor, delta: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.map_from_prior(noise, delta, condition)
