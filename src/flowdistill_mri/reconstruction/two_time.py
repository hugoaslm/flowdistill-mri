from __future__ import annotations

from typing import Protocol

import torch


class TwoTimeFlowMap(Protocol):
    """Track B interface. This is FreeFlow-inspired, not paper-faithful FreeFlow."""

    def map(
        self,
        state: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        measurement: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


def train_two_time_flow_map(*args, **kwargs):
    raise NotImplementedError(
        "Track B training is intentionally disabled during repository initialization"
    )
