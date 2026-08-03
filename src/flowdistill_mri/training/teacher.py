from __future__ import annotations

import json
from pathlib import Path

import torch

from flowdistill_mri.config import Config
from flowdistill_mri.flows.interpolation import rectified_flow_pair
from flowdistill_mri.training.builders import build_dataset, build_teacher, infinite_batches
from flowdistill_mri.training.checkpoint import load_checkpoint, save_checkpoint
from flowdistill_mri.training.ema import EMA
from flowdistill_mri.training.runtime import select_device, set_seed


def teacher_step(
    teacher: torch.nn.Module, optimizer: torch.optim.Optimizer, data: torch.Tensor
) -> torch.Tensor:
    noise = torch.randn_like(data)
    time = torch.rand(data.shape[0], device=data.device, dtype=data.dtype)
    state, target = rectified_flow_pair(data, noise, time)
    prediction = teacher.velocity(state, time)
    loss = (prediction - target).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


def train_teacher(cfg: Config, output: str | Path, *, resume: bool = False) -> Path:
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    teacher = build_teacher(cfg).to(device)
    ema = EMA(teacher.backbone, cfg.training.ema_decay)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=cfg.training.learning_rate)
    output = Path(output)
    start = 0
    if resume:
        start, manifest = load_checkpoint(
            output,
            {"teacher": teacher, "teacher_ema": ema.shadow},
            {"optimizer": optimizer},
            expected={
                "model_type": "rectified_flow_teacher",
                "time_convention": "t0_noise_t1_data",
            },
        )
        if manifest.get("config") != json.loads(json.dumps(cfg.to_dict())):
            raise ValueError("resume config differs from the checkpoint config")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty checkpoint directory: {output}")
    batches = infinite_batches(build_dataset(cfg), cfg.training.batch_size, cfg.seed)
    for _ in range(start):
        next(batches)
    if start >= cfg.training.teacher_steps:
        return output
    loss = torch.tensor(float("nan"))
    for _ in range(start, cfg.training.teacher_steps):
        loss = teacher_step(teacher, optimizer, next(batches).to(device))
        ema.update(teacher.backbone)
    save_checkpoint(
        output,
        {"teacher": teacher, "teacher_ema": ema.shadow},
        cfg.training.teacher_steps,
        {
            "model_type": "rectified_flow_teacher",
            "time_convention": "t0_noise_t1_data",
            "final_loss": float(loss),
            "config": cfg.to_dict(),
        },
        {"optimizer": optimizer},
    )
    return output
