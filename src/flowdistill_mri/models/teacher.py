from __future__ import annotations

import torch
from torch import nn

from flowdistill_mri.flows.solvers import integrate


class RectifiedFlowTeacher(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def velocity(
        self, x_t: torch.Tensor, time: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.backbone(x_t, time, condition)

    def forward(
        self, x_t: torch.Tensor, time: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.velocity(x_t, time, condition)

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        schedule: torch.Tensor,
        solver: str = "heun",
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return integrate(
            lambda state, time: self.velocity(state, time, condition), noise, schedule, solver
        )

