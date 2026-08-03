from __future__ import annotations

import torch


def cartesian_mask(
    shape: tuple[int, int], acceleration: int, center_fraction: float, seed: int
) -> torch.Tensor:
    """Return a deterministic 1-D Cartesian mask broadcast across rows."""
    height, width = shape
    if acceleration < 2 or not 0 < center_fraction < 1:
        raise ValueError("invalid acceleration or center fraction")
    target = max(1, round(width / acceleration))
    center = min(target, max(1, round(width * center_fraction)))
    start = (width - center) // 2
    chosen = torch.zeros(width, dtype=torch.bool)
    chosen[start : start + center] = True
    remaining = target - center
    candidates = torch.where(~chosen)[0]
    if remaining > 0:
        generator = torch.Generator().manual_seed(seed)
        picked = candidates[torch.randperm(len(candidates), generator=generator)[:remaining]]
        chosen[picked] = True
    return chosen.unsqueeze(0).expand(height, -1).clone()

