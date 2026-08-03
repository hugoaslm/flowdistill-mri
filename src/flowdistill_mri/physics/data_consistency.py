from __future__ import annotations

import torch

from flowdistill_mri.physics.fft import fft2c, ifft2c
from flowdistill_mri.physics.operators import expand_mask


def hard_data_consistency(
    image: torch.Tensor, measured_kspace: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    predicted = fft2c(image)
    selected = expand_mask(mask, predicted)
    return ifft2c(predicted * (1.0 - selected) + measured_kspace * selected)


def soft_data_consistency(
    image: torch.Tensor,
    measured_kspace: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if not 0 <= weight <= 1:
        raise ValueError("soft data-consistency weight must be in [0, 1]")
    predicted = fft2c(image)
    selected = expand_mask(mask, predicted)
    blended = predicted + selected * weight * (measured_kspace - predicted)
    return ifft2c(blended)

