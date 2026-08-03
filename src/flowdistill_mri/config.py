from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    kind: str = "synthetic"
    samples: int = 16
    resolution: int = 32
    acceleration: int = 4
    center_fraction: float = 0.125


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 8
    channel_multipliers: tuple[int, ...] = (1, 2)
    residual_blocks: int = 1
    time_dim: int = 32


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation: int = 1
    teacher_steps: int = 4
    distillation_steps: int = 4
    learning_rate: float = 1e-3
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    ema_decay: float = 0.99


@dataclass(frozen=True)
class DistillationConfig:
    intervals: int = 4
    prediction_probability: float = 0.75
    correction_weight: float = 0.3
    correction_delay: float = 0.1
    correction_warmup: float = 0.1
    norm_exponent: float = 1.0


@dataclass(frozen=True)
class SamplingConfig:
    teacher_steps: int = 4
    solver: str = "heun"


@dataclass(frozen=True)
class TrackBConfig:
    enabled: bool = False


@dataclass(frozen=True)
class Config:
    experiment: str
    seed: int = 42
    device: str = "cpu"
    output_root: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    track_b: TrackBConfig = field(default_factory=TrackBConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(cls: type, raw: dict[str, Any], name: str):
    values = dict(raw.get(name, {}))
    if cls is ModelConfig and "channel_multipliers" in values:
        values["channel_multipliers"] = tuple(values["channel_multipliers"])
    return cls(**values)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = Config(
        experiment=str(raw["experiment"]),
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "cpu")),
        output_root=str(raw.get("output_root", "outputs")),
        data=_section(DataConfig, raw, "data"),
        model=_section(ModelConfig, raw, "model"),
        training=_section(TrainingConfig, raw, "training"),
        distillation=_section(DistillationConfig, raw, "distillation"),
        sampling=_section(SamplingConfig, raw, "sampling"),
        track_b=_section(TrackBConfig, raw, "track_b"),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.data.kind not in {"synthetic", "fastmri"}:
        raise ValueError("data.kind must be 'synthetic' or 'fastmri'")
    downsample_factor = 2 ** (len(cfg.model.channel_multipliers) - 1)
    if cfg.data.resolution < 16 or cfg.data.resolution % downsample_factor:
        raise ValueError("resolution must be divisible by every U-Net downsampling level")
    if cfg.data.acceleration < 2:
        raise ValueError("acceleration must be at least 2")
    if not 0 < cfg.data.center_fraction < 1:
        raise ValueError("center_fraction must be in (0, 1)")
    if cfg.model.base_channels < 4 or cfg.model.time_dim < 8:
        raise ValueError("model dimensions are too small")
    if cfg.training.batch_size < 1 or cfg.training.gradient_accumulation < 1:
        raise ValueError("batch and accumulation sizes must be positive")
    if cfg.training.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if cfg.distillation.intervals < 1:
        raise ValueError("distillation.intervals must be positive")
    if cfg.sampling.solver not in {"euler", "heun"}:
        raise ValueError("sampling.solver must be euler or heun")
