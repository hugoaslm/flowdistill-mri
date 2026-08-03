import pytest
import torch
from torch import nn

from flowdistill_mri.distillation.freeflow import (
    correction_losses,
    correction_schedule,
    prediction_loss,
)
from flowdistill_mri.models.freeflow import PriorAnchoredFreeFlow


class ScalarBackbone(nn.Module):
    def __init__(self, value=0.25):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.times = []

    def forward(self, x, time, condition=None):
        self.times.append(time.detach().clone())
        return torch.ones_like(x) * self.value


class ScalarTeacher(nn.Module):
    def __init__(self, value=0.25):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.times = []

    def velocity(self, x, time, condition=None):
        self.times.append(time.detach().clone())
        return torch.ones_like(x) * self.value


def test_prediction_matches_constant_teacher_and_freezes_teacher():
    student = PriorAnchoredFreeFlow(ScalarBackbone())
    teacher = ScalarTeacher()
    noise = torch.randn(2, 2, 4, 4)
    loss, metrics = prediction_loss(
        student, teacher, noise, intervals=4, interval_index=torch.tensor([0, 2])
    )
    assert metrics.residual_mse < 1e-10
    loss.backward()
    assert student.backbone.value.grad is not None
    assert teacher.value.grad is None
    assert torch.allclose(teacher.times[0], torch.tensor([0.0, 0.5]))


def test_correction_time_conversion_and_gradient_boundaries():
    student = PriorAnchoredFreeFlow(ScalarBackbone())
    teacher = ScalarTeacher()
    corrector = ScalarBackbone()
    noise = torch.randn(2, 2, 4, 4)
    auxiliary, student_loss, metrics = correction_losses(
        student,
        teacher,
        corrector,
        noise,
        step_ratio=0.5,
        renoise_level=torch.full((2,), 0.25),
    )
    assert torch.allclose(teacher.times[0], torch.full((2,), 0.75))
    assert metrics.schedule_weight == 1.0
    auxiliary.backward()
    assert corrector.value.grad is not None
    assert student.backbone.value.grad is None
    corrector.zero_grad()
    student_loss.backward()
    assert student.backbone.value.grad is not None
    assert corrector.value.grad is None
    assert teacher.value.grad is None


def test_correction_schedule():
    assert correction_schedule(0.05, 0.1, 0.1) == 0.0
    assert correction_schedule(0.15, 0.1, 0.1) == pytest.approx(0.5)
    assert correction_schedule(0.25, 0.1, 0.1) == 1.0
