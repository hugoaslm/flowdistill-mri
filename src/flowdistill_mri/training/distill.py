from __future__ import annotations

import json
from pathlib import Path

import torch

from flowdistill_mri.config import Config
from flowdistill_mri.distillation.freeflow import correction_losses, prediction_loss
from flowdistill_mri.models.freeflow import PriorAnchoredFreeFlow
from flowdistill_mri.training.builders import (
    build_backbone,
    build_teacher,
)
from flowdistill_mri.training.checkpoint import load_checkpoint, save_checkpoint
from flowdistill_mri.training.ema import EMA
from flowdistill_mri.training.runtime import select_device, set_seed


def distillation_step(
    student: PriorAnchoredFreeFlow,
    teacher: torch.nn.Module,
    corrector: torch.nn.Module,
    student_optimizer: torch.optim.Optimizer,
    corrector_optimizer: torch.optim.Optimizer,
    noise: torch.Tensor,
    cfg: Config,
    step_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = noise.shape[0]
    prediction_count = max(1, min(batch - 1, round(batch * cfg.distillation.prediction_probability)))
    prediction_noise = noise if batch == 1 else noise[:prediction_count]
    correction_noise = noise if batch == 1 else noise[prediction_count:]
    prediction, _ = prediction_loss(
        student,
        teacher,
        prediction_noise,
        intervals=cfg.distillation.intervals,
        norm_exponent=cfg.distillation.norm_exponent,
    )
    auxiliary, correction, _ = correction_losses(
        student,
        teacher,
        corrector,
        correction_noise,
        step_ratio=step_ratio,
        correction_weight=cfg.distillation.correction_weight,
        delay=cfg.distillation.correction_delay,
        warmup=cfg.distillation.correction_warmup,
    )
    student_optimizer.zero_grad(set_to_none=True)
    corrector_optimizer.zero_grad(set_to_none=True)
    auxiliary.backward()
    (prediction + correction).backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
    corrector_optimizer.step()
    student_optimizer.step()
    return prediction.detach(), auxiliary.detach()


def distill_freeflow(
    cfg: Config,
    teacher_checkpoint: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
) -> Path:
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    teacher = build_teacher(cfg).to(device)
    load_checkpoint(
        teacher_checkpoint,
        {"teacher": teacher},
        expected={"model_type": "rectified_flow_teacher", "time_convention": "t0_noise_t1_data"},
        restore_rng=False,
    )
    teacher.eval().requires_grad_(False)
    student = PriorAnchoredFreeFlow(build_backbone(cfg).to(device))
    corrector = build_backbone(cfg).to(device)
    student.backbone.load_state_dict(teacher.backbone.state_dict())
    corrector.load_state_dict(teacher.backbone.state_dict())
    ema = EMA(student.backbone, cfg.training.ema_decay)
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.training.learning_rate)
    corrector_optimizer = torch.optim.AdamW(corrector.parameters(), lr=cfg.training.learning_rate)
    output = Path(output)
    start = 0
    if resume:
        start, manifest = load_checkpoint(
            output,
            {"student": student, "student_ema": ema.shadow, "corrector": corrector},
            {"student": student_optimizer, "corrector": corrector_optimizer},
            expected={
                "model_type": "prior_anchored_freeflow",
                "track": "A",
                "target_free": True,
                "time_convention": "t0_noise_t1_data",
            },
        )
        if manifest.get("config") != json.loads(json.dumps(cfg.to_dict())):
            raise ValueError("resume config differs from the checkpoint config")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty checkpoint directory: {output}")
    if start >= cfg.training.distillation_steps:
        return output
    prediction = auxiliary = torch.tensor(float("nan"))
    total = cfg.training.distillation_steps
    for step in range(start, total):
        noise = torch.randn(
            cfg.training.batch_size,
            2,
            cfg.data.resolution,
            cfg.data.resolution,
            device=device,
        )
        prediction, auxiliary = distillation_step(
            student,
            teacher,
            corrector,
            student_optimizer,
            corrector_optimizer,
            noise,
            cfg,
            step / max(total, 1),
        )
        ema.update(student.backbone)
    save_checkpoint(
        output,
        {"student": student, "student_ema": ema.shadow, "corrector": corrector},
        total,
        {
            "model_type": "prior_anchored_freeflow",
            "track": "A",
            "target_free": True,
            "time_convention": "t0_noise_t1_data",
            "final_prediction_loss": float(prediction),
            "final_auxiliary_loss": float(auxiliary),
            "config": cfg.to_dict(),
        },
        {"student": student_optimizer, "corrector": corrector_optimizer},
    )
    return output
