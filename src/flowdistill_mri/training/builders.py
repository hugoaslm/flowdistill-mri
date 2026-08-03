from __future__ import annotations

import torch

from flowdistill_mri.config import Config
from flowdistill_mri.data.synthetic import SyntheticMRIDataset
from flowdistill_mri.models.teacher import RectifiedFlowTeacher
from flowdistill_mri.models.unet import TimeConditionedUNet


def build_backbone(cfg: Config) -> TimeConditionedUNet:
    return TimeConditionedUNet(
        base_channels=cfg.model.base_channels,
        channel_multipliers=cfg.model.channel_multipliers,
        residual_blocks=cfg.model.residual_blocks,
        time_dim=cfg.model.time_dim,
    )


def build_teacher(cfg: Config) -> RectifiedFlowTeacher:
    return RectifiedFlowTeacher(build_backbone(cfg))


def build_dataset(cfg: Config) -> SyntheticMRIDataset:
    if cfg.data.kind != "synthetic":
        raise NotImplementedError(
            "fastMRI loading is scheduled for the real-data phase; use a synthetic smoke config"
        )
    return SyntheticMRIDataset(cfg.data.samples, cfg.data.resolution, cfg.seed)


def infinite_batches(dataset, batch_size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(len(dataset), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if len(indices) < batch_size:
                continue
            yield torch.stack([dataset[int(index)] for index in indices])

