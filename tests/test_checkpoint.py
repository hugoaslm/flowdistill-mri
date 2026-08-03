import torch
from torch import nn

from flowdistill_mri.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_restores_weights_optimizer_and_rng(tmp_path):
    torch.manual_seed(9)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    saved_weight = model.weight.detach().clone()
    save_checkpoint(
        tmp_path, {"model": model}, 7, {"model_type": "unit"}, {"optimizer": optimizer}
    )
    expected_random = torch.randn(4)
    with torch.no_grad():
        model.weight.zero_()
    torch.manual_seed(100)
    step, manifest = load_checkpoint(
        tmp_path,
        {"model": model},
        {"optimizer": optimizer},
        expected={"model_type": "unit"},
    )
    assert step == 7 and manifest["model_type"] == "unit"
    assert torch.equal(model.weight, saved_weight)
    assert torch.equal(torch.randn(4), expected_random)

