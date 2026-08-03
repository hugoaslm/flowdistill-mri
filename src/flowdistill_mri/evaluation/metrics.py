from __future__ import annotations

import torch


def nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().sum() / target.square().sum().clamp_min(1e-12)


def complex_relative_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).flatten(1).norm(dim=1).mean() / target.flatten(1).norm(
        dim=1
    ).mean().clamp_min(1e-12)


def magnitude(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum(dim=-3).sqrt()


def magnitude_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mag, target_mag = magnitude(prediction), magnitude(target)
    error = (pred_mag - target_mag).square().mean().clamp_min(1e-12)
    peak = target_mag.amax().clamp_min(1e-12)
    return 20 * torch.log10(peak) - 10 * torch.log10(error)
