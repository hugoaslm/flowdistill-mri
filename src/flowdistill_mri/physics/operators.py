from __future__ import annotations

from dataclasses import dataclass

import torch

from flowdistill_mri.physics.fft import fft2c, ifft2c


def expand_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = mask
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    while value.ndim < reference.ndim:
        value = value.unsqueeze(0)
    return value.to(device=reference.device, dtype=reference.dtype)


@dataclass(frozen=True)
class SingleCoilMRI:
    mask: torch.Tensor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        kspace = fft2c(image)
        return kspace * expand_mask(self.mask, kspace)

    def adjoint(self, measured_kspace: torch.Tensor) -> torch.Tensor:
        return ifft2c(measured_kspace * expand_mask(self.mask, measured_kspace))

    def normal(self, image: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(image))
