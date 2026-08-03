from __future__ import annotations

import torch


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert [..., 2, H, W] real/imag channels to native complex."""
    if x.ndim < 3 or x.shape[-3] != 2:
        raise ValueError(f"expected channel dimension of size 2 at -3, got {tuple(x.shape)}")
    return torch.complex(x.select(-3, 0), x.select(-3, 1))


def from_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert native complex [..., H, W] to [..., 2, H, W]."""
    if not torch.is_complex(x):
        raise TypeError("from_complex expects a native complex tensor")
    return torch.stack((x.real, x.imag), dim=-3)

