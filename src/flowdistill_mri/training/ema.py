from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        self.shadow.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for name, value in self.shadow.state_dict().items():
            source = model_state[name].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)

