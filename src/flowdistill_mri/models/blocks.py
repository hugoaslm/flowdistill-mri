from __future__ import annotations

import math

import torch
from torch import nn


def group_count(channels: int) -> int:
    for count in (8, 4, 2):
        if channels % count == 0:
            return count
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.reshape(-1).float()
        half = self.dimension // 2
        scale = math.log(10_000) / max(half - 1, 1)
        frequencies = torch.exp(-scale * torch.arange(half, device=time.device))
        embedding = time[:, None] * frequencies[None]
        result = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        if result.shape[-1] < self.dimension:
            result = torch.nn.functional.pad(result, (0, 1))
        return result


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        hidden = hidden + self.time(time)[:, :, None, None]
        hidden = self.conv2(torch.nn.functional.silu(self.norm2(hidden)))
        return hidden + self.skip(x)

