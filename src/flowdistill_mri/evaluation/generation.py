from __future__ import annotations

import json
from pathlib import Path

import torch

from flowdistill_mri.config import Config
from flowdistill_mri.evaluation.metrics import complex_relative_error, magnitude_psnr, nmse
from flowdistill_mri.models.freeflow import PriorAnchoredFreeFlow
from flowdistill_mri.training.builders import build_backbone, build_teacher
from flowdistill_mri.training.checkpoint import load_checkpoint
from flowdistill_mri.training.runtime import select_device, set_seed


@torch.no_grad()
def evaluate_generation(
    cfg: Config,
    teacher_checkpoint: str | Path,
    student_checkpoint: str | Path,
    output: str | Path,
) -> dict[str, float]:
    print("[evaluation] loading teacher and student checkpoints", flush=True)
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    teacher = build_teacher(cfg).to(device).eval()
    student = PriorAnchoredFreeFlow(build_backbone(cfg)).to(device).eval()
    load_checkpoint(teacher_checkpoint, {"teacher": teacher}, restore_rng=False)
    load_checkpoint(student_checkpoint, {"student": student}, restore_rng=False)
    noise = torch.randn(4, 2, cfg.data.resolution, cfg.data.resolution, device=device)
    print("[evaluation] generating 4 paired samples", flush=True)
    schedule = torch.linspace(0, 1, cfg.sampling.teacher_steps + 1, device=device)
    target = teacher.sample(noise, schedule, cfg.sampling.solver)
    prediction = student.map_from_prior(noise, torch.ones(4, device=device))
    results = {
        "nmse": float(nmse(prediction, target)),
        "complex_relative_error": float(complex_relative_error(prediction, target)),
        "magnitude_psnr": float(magnitude_psnr(prediction, target)),
        "teacher_nfe": float(
            cfg.sampling.teacher_steps * (2 if cfg.sampling.solver == "heun" else 1)
        ),
        "student_nfe": 1.0,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[evaluation] metrics saved: {output / 'generation_metrics.json'}", flush=True)
    return results
