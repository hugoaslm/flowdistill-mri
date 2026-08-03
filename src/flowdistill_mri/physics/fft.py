from __future__ import annotations

import torch

from flowdistill_mri.physics.complex import from_complex, to_complex


def _shift(x: torch.Tensor, inverse: bool) -> torch.Tensor:
    function = torch.fft.ifftshift if inverse else torch.fft.fftshift
    return function(x, dim=(-2, -1))


def fft2c(x: torch.Tensor) -> torch.Tensor:
    value = to_complex(x)
    value = _shift(value, inverse=True)
    value = torch.fft.fft2(value, dim=(-2, -1), norm="ortho")
    return from_complex(_shift(value, inverse=False))


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    value = to_complex(kspace)
    value = _shift(value, inverse=True)
    value = torch.fft.ifft2(value, dim=(-2, -1), norm="ortho")
    return from_complex(_shift(value, inverse=False))

