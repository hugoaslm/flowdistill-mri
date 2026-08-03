from __future__ import annotations

import torch


def append_dims(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = time
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def rectified_flow_pair(
    data: torch.Tensor, noise: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the repository convention: t=0 noise, t=1 data."""
    if data.shape != noise.shape:
        raise ValueError("data and noise must have identical shapes")
    expanded = append_dims(time, data)
    state = (1.0 - expanded) * noise + expanded * data
    return state, data - noise

