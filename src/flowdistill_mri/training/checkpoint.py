from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    directory: str | Path,
    models: dict[str, torch.nn.Module],
    step: int,
    manifest: dict[str, Any],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights: dict[str, torch.Tensor] = {}
    for model_name, model in models.items():
        for name, value in model.state_dict().items():
            weights[f"{model_name}.{name}"] = value.detach().cpu().contiguous()
    weight_tmp = directory / ".weights.safetensors.tmp"
    save_file(weights, str(weight_tmp))
    os.replace(weight_tmp, directory / "weights.safetensors")
    state_tmp = directory / ".trainer_state.pt.tmp"
    torch.save(
        {
            "step": step,
            "optimizers": {name: optimizer.state_dict() for name, optimizer in (optimizers or {}).items()},
            "rng": rng_state(),
        },
        state_tmp,
    )
    os.replace(state_tmp, directory / "trainer_state.pt")
    manifest_tmp = directory / ".manifest.json.tmp"
    manifest_tmp.write_text(json.dumps({"step": step, **manifest}, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, directory / "manifest.json")


def load_checkpoint(
    directory: str | Path,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    expected: dict[str, Any] | None = None,
    restore_rng: bool = True,
) -> tuple[int, dict[str, Any]]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for key, value in (expected or {}).items():
        if manifest.get(key) != value:
            raise ValueError(f"checkpoint mismatch for {key}: {manifest.get(key)!r} != {value!r}")
    weights = load_file(str(directory / "weights.safetensors"))
    for model_name, model in models.items():
        prefix = f"{model_name}."
        selected = {name[len(prefix) :]: value for name, value in weights.items() if name.startswith(prefix)}
        if not selected:
            raise KeyError(f"checkpoint has no model named {model_name!r}")
        model.load_state_dict(selected)
    state = torch.load(directory / "trainer_state.pt", map_location="cpu", weights_only=False)
    for name, optimizer in (optimizers or {}).items():
        if name in state["optimizers"]:
            optimizer.load_state_dict(state["optimizers"][name])
    if restore_rng:
        restore_rng_state(state["rng"])
    return int(state["step"]), manifest

