import torch

from flowdistill_mri.data.masks import cartesian_mask
from flowdistill_mri.physics.complex import from_complex, to_complex
from flowdistill_mri.physics.data_consistency import hard_data_consistency, soft_data_consistency
from flowdistill_mri.physics.fft import fft2c, ifft2c
from flowdistill_mri.physics.operators import SingleCoilMRI


def test_complex_and_fft_round_trips():
    torch.manual_seed(0)
    image = torch.randn(3, 2, 16, 16)
    assert torch.equal(from_complex(to_complex(image)), image)
    assert torch.allclose(ifft2c(fft2c(image)), image, atol=1e-5, rtol=1e-5)


def test_cartesian_mask_is_deterministic_and_has_requested_count():
    first = cartesian_mask((16, 32), acceleration=4, center_fraction=0.125, seed=7)
    second = cartesian_mask((16, 32), acceleration=4, center_fraction=0.125, seed=7)
    assert torch.equal(first, second)
    assert int(first[0].sum()) == 8
    assert torch.equal(first[0], first[-1])


def test_forward_adjoint_and_data_consistency():
    torch.manual_seed(1)
    target = torch.randn(2, 2, 16, 16)
    estimate = torch.randn_like(target)
    mask = cartesian_mask((16, 16), 4, 0.125, 3)
    operator = SingleCoilMRI(mask)
    measured = operator.forward(target)
    assert operator.adjoint(measured).shape == target.shape
    projected = hard_data_consistency(estimate, measured, mask)
    selected = mask[None, None]
    assert torch.allclose(fft2c(projected) * selected, measured, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        soft_data_consistency(estimate, measured, mask, 1.0),
        projected,
        atol=1e-5,
        rtol=1e-5,
    )
