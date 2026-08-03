from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset


def make_phantom(resolution: int, seed: int) -> torch.Tensor:
    """Create a deterministic complex ellipse phantom for infrastructure tests."""
    generator = torch.Generator().manual_seed(seed)
    axis = torch.linspace(-1.0, 1.0, resolution)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    magnitude = torch.zeros_like(xx)
    phase = torch.zeros_like(xx)
    for _ in range(5):
        center = torch.rand(2, generator=generator) * 1.2 - 0.6
        radii = torch.rand(2, generator=generator) * 0.3 + 0.1
        angle = float(torch.rand((), generator=generator) * math.pi)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x_rot = cos_a * (xx - center[0]) + sin_a * (yy - center[1])
        y_rot = -sin_a * (xx - center[0]) + cos_a * (yy - center[1])
        ellipse = (x_rot / radii[0]).square() + (y_rot / radii[1]).square() <= 1
        magnitude = magnitude + ellipse * (0.15 + torch.rand((), generator=generator) * 0.5)
        phase = phase + ellipse * (torch.rand((), generator=generator) - 0.5)
    magnitude = magnitude / magnitude.amax().clamp_min(1e-6)
    return torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)))


class SyntheticMRIDataset(Dataset[torch.Tensor]):
    def __init__(self, samples: int, resolution: int, seed: int = 0) -> None:
        self.samples = samples
        self.resolution = resolution
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.samples:
            raise IndexError(index)
        return make_phantom(self.resolution, self.seed + index)
