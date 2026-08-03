import torch

from flowdistill_mri.config import load_config
from flowdistill_mri.models.freeflow import PriorAnchoredFreeFlow
from flowdistill_mri.training.builders import build_backbone, build_teacher
from flowdistill_mri.training.distill import distillation_step
from flowdistill_mri.training.teacher import teacher_step


def test_one_teacher_and_distillation_step_have_finite_losses():
    torch.manual_seed(3)
    cfg = load_config("configs/smoke_ci.yaml")
    teacher = build_teacher(cfg)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    data = torch.randn(2, 2, 32, 32)
    loss = teacher_step(teacher, teacher_optimizer, data)
    assert torch.isfinite(loss)
    teacher.zero_grad(set_to_none=True)
    teacher.eval().requires_grad_(False)
    student = PriorAnchoredFreeFlow(build_backbone(cfg))
    corrector = build_backbone(cfg)
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    corrector_optimizer = torch.optim.AdamW(corrector.parameters(), lr=1e-3)
    prediction, auxiliary = distillation_step(
        student,
        teacher,
        corrector,
        student_optimizer,
        corrector_optimizer,
        torch.randn_like(data),
        cfg,
        0.5,
    )
    assert torch.isfinite(prediction) and torch.isfinite(auxiliary)
    assert all(parameter.grad is None for parameter in teacher.parameters())
