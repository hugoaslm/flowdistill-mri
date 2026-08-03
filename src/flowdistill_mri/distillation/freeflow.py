from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flowdistill_mri.flows.interpolation import append_dims
from flowdistill_mri.models.freeflow import PriorAnchoredFreeFlow


@dataclass(frozen=True)
class PredictionMetrics:
    loss: float
    residual_mse: float


@dataclass(frozen=True)
class CorrectionMetrics:
    auxiliary_loss: float
    student_loss: float
    adaptive_weight: float
    schedule_weight: float


def prediction_loss(
    student: PriorAnchoredFreeFlow,
    teacher: nn.Module,
    noise: torch.Tensor,
    intervals: int = 8,
    norm_exponent: float = 1.0,
    interval_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, PredictionMetrics]:
    """Discrete FreeFlow prediction objective under t=0 noise, t=1 data."""
    batch = noise.shape[0]
    step = 1.0 / intervals
    if interval_index is None:
        interval_index = torch.randint(0, intervals, (batch,), device=noise.device)
    delta = interval_index.to(noise.dtype) * step
    next_delta = delta + step
    velocity = student.average_velocity(noise, delta)
    next_velocity = student.average_velocity(noise, next_delta)
    predicted_state = noise + append_dims(delta, noise) * velocity
    with torch.no_grad():
        teacher_velocity = teacher.velocity(predicted_state.detach(), delta)
        finite_difference = append_dims(delta, noise) * (
            next_velocity.detach() - velocity.detach()
        ) / step
        stopped_target_term = finite_difference - teacher_velocity
    residual = next_velocity + stopped_target_term
    per_sample = residual.square().flatten(1).mean(1)
    weights = (per_sample.detach() + 1e-4).pow(-norm_exponent)
    loss = (weights * per_sample).mean()
    return loss, PredictionMetrics(float(loss.detach()), float(per_sample.detach().mean()))


def correction_schedule(step_ratio: float, delay: float, warmup: float) -> float:
    if step_ratio < delay:
        return 0.0
    if warmup > 0 and step_ratio < delay + warmup:
        return (step_ratio - delay) / warmup
    return 1.0


def correction_losses(
    student: PriorAnchoredFreeFlow,
    teacher: nn.Module,
    corrector: nn.Module,
    noise: torch.Tensor,
    *,
    step_ratio: float,
    correction_weight: float = 0.3,
    delay: float = 0.1,
    warmup: float = 0.1,
    renoise_level: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, CorrectionMetrics]:
    """FreeFlow auxiliary noising model and stopped student correction gradient."""
    batch = noise.shape[0]
    ones = torch.ones(batch, device=noise.device, dtype=noise.dtype)
    endpoint = student.map_from_prior(noise, ones)
    independent_noise = torch.randn_like(endpoint)
    if renoise_level is None:
        logits = torch.randn(batch, device=noise.device, dtype=noise.dtype) * 1.6 + 0.8
        renoise_level = torch.sigmoid(logits)
    # r=0 is generated data and r=1 is noise; repository teacher time is therefore 1-r.
    teacher_time = 1.0 - renoise_level
    renoised = (1.0 - append_dims(renoise_level, endpoint)) * endpoint + append_dims(
        renoise_level, endpoint
    ) * independent_noise

    detached_endpoint = endpoint.detach()
    auxiliary_state = (1.0 - append_dims(renoise_level, endpoint)) * detached_endpoint + append_dims(
        renoise_level, endpoint
    ) * independent_noise
    fake_velocity = corrector(auxiliary_state, teacher_time)
    auxiliary_target = detached_endpoint - independent_noise
    auxiliary_loss = (fake_velocity - auxiliary_target).square().mean()

    with torch.no_grad():
        teacher_velocity = teacher.velocity(renoised.detach(), teacher_time)
        fake_velocity_at_state = corrector(renoised.detach(), teacher_time)
        correction_gap = (fake_velocity_at_state - teacher_velocity).flatten(1).norm(dim=1).mean()
        probe_step = 1.0 / 8.0
        delta = torch.rand(batch, device=noise.device, dtype=noise.dtype) * (1.0 - probe_step)
        start = student.map_from_prior(noise, delta)
        end = student.map_from_prior(noise, delta + probe_step)
        generating_velocity = (end - start) / probe_step
        prediction_gap = (
            generating_velocity - teacher.velocity(start, delta)
        ).flatten(1).norm(dim=1).mean()
        adaptive = correction_weight * prediction_gap / (correction_gap + 1e-6)

    schedule = correction_schedule(step_ratio, delay, warmup)
    discrepancy = (fake_velocity_at_state - teacher_velocity).detach()
    terminal_velocity = student.average_velocity(noise, ones)
    raw_student_loss = (terminal_velocity * discrepancy).flatten(1).mean(1).mean()
    student_loss = float(adaptive) * schedule * raw_student_loss
    metrics = CorrectionMetrics(
        auxiliary_loss=float(auxiliary_loss.detach()),
        student_loss=float(raw_student_loss.detach()),
        adaptive_weight=float(adaptive),
        schedule_weight=schedule,
    )
    return auxiliary_loss, student_loss, metrics

