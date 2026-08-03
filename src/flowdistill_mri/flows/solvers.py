from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import torch

Velocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def integrate(
    velocity: Velocity,
    initial: torch.Tensor,
    schedule: torch.Tensor,
    solver: str = "heun",
) -> torch.Tensor:
    if schedule.ndim != 1 or len(schedule) < 2:
        raise ValueError("schedule must be a one-dimensional tensor with at least two points")
    if not bool(torch.all(schedule[1:] > schedule[:-1])):
        raise ValueError("schedule must be strictly increasing")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be euler or heun")
    state = initial
    batch = state.shape[0]
    for start, end in pairwise(schedule):
        dt = end - start
        start_batch = start.expand(batch).to(device=state.device, dtype=state.dtype)
        first = velocity(state, start_batch)
        proposal = state + dt * first
        if solver == "euler":
            state = proposal
        else:
            end_batch = end.expand(batch).to(device=state.device, dtype=state.dtype)
            state = state + 0.5 * dt * (first + velocity(proposal, end_batch))
    return state
