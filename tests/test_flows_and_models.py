import torch
from torch import nn

from flowdistill_mri.flows.interpolation import rectified_flow_pair
from flowdistill_mri.flows.solvers import integrate
from flowdistill_mri.models.freeflow import PriorAnchoredFreeFlow
from flowdistill_mri.models.unet import TimeConditionedUNet


def test_rectified_flow_endpoints():
    data = torch.randn(2, 2, 4, 4)
    noise = torch.randn_like(data)
    at_noise, velocity = rectified_flow_pair(data, noise, torch.zeros(2))
    at_data, _ = rectified_flow_pair(data, noise, torch.ones(2))
    assert torch.equal(at_noise, noise)
    assert torch.equal(at_data, data)
    assert torch.equal(velocity, data - noise)


def test_solvers_integrate_constant_velocity():
    initial = torch.zeros(2, 2, 4, 4)
    schedule = torch.linspace(0, 1, 5)

    def velocity(state, time):
        return torch.ones_like(state) * 2

    expected = torch.ones_like(initial) * 2
    assert torch.allclose(integrate(velocity, initial, schedule, "euler"), expected)
    assert torch.allclose(integrate(velocity, initial, schedule, "heun"), expected)


class OnesBackbone(nn.Module):
    def forward(self, x, time, condition=None):
        return torch.ones_like(x)


def test_prior_anchored_map_identity_and_displacement():
    model = PriorAnchoredFreeFlow(OnesBackbone())
    noise = torch.randn(2, 2, 4, 4)
    assert torch.equal(model.map_from_prior(noise, torch.zeros(2)), noise)
    assert torch.allclose(model.map_from_prior(noise, torch.ones(2)), noise + 1)


def test_compact_unet_preserves_shape():
    model = TimeConditionedUNet(base_channels=8, channel_multipliers=(1, 2), time_dim=32)
    output = model(torch.randn(2, 2, 32, 32), torch.tensor([0.0, 1.0]))
    assert output.shape == (2, 2, 32, 32)
    assert torch.isfinite(output).all()


def test_research_unet_supports_multiple_residual_blocks():
    model = TimeConditionedUNet(
        base_channels=8, channel_multipliers=(1, 2), residual_blocks=2, time_dim=32
    )
    output = model(torch.randn(1, 2, 16, 16), torch.tensor([0.5]))
    assert output.shape == (1, 2, 16, 16)
