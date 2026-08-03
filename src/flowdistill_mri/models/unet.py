from __future__ import annotations

import torch
from torch import nn

from flowdistill_mri.models.blocks import ResidualBlock, SinusoidalTimeEmbedding, group_count


class TimeConditionedUNet(nn.Module):
    """Compact 2-D U-Net shared by teacher, Track A student, and corrector."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 2),
        residual_blocks: int = 1,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")
        channels = [base_channels * value for value in channel_multipliers]
        self.input = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = channels[0]
        for index, output in enumerate(channels):
            level = nn.ModuleList([ResidualBlock(current, output, time_dim)])
            level.extend(ResidualBlock(output, output, time_dim) for _ in range(residual_blocks - 1))
            self.down_blocks.append(level)
            current = output
            if index < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(current, current, 3, stride=2, padding=1))
        self.middle = ResidualBlock(current, current, time_dim)
        self.up_blocks = nn.ModuleList()
        for skip_channels in reversed(channels[:-1]):
            level = nn.ModuleList([ResidualBlock(current + skip_channels, skip_channels, time_dim)])
            level.extend(
                ResidualBlock(skip_channels, skip_channels, time_dim)
                for _ in range(residual_blocks - 1)
            )
            self.up_blocks.append(level)
            current = skip_channels
        self.output_norm = nn.GroupNorm(group_count(current), current)
        self.output = nn.Conv2d(current, out_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, time: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        if condition is not None:
            raise NotImplementedError("conditional modeling is outside the initial Track A scope")
        if time.ndim == 0:
            time = time.expand(x.shape[0])
        embedding = self.time_embedding(time)
        hidden = self.input(x)
        skips: list[torch.Tensor] = []
        for index, level in enumerate(self.down_blocks):
            for block in level:
                hidden = block(hidden, embedding)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        hidden = self.middle(hidden, embedding)
        for level, skip in zip(self.up_blocks, reversed(skips[:-1])):
            hidden = torch.nn.functional.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = torch.cat((hidden, skip), dim=1)
            for block in level:
                hidden = block(hidden, embedding)
        return self.output(torch.nn.functional.silu(self.output_norm(hidden)))
