from __future__ import annotations

import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from flowdistill_mri.config import Config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select a smoke CPU config instead")
    return torch.device(requested)


def create_run_directory(cfg: Config, stage: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(cfg.output_root) / f"{cfg.experiment}-{stage}-{timestamp}"
    suffix = 1
    while path.exists():
        path = Path(cfg.output_root) / f"{cfg.experiment}-{stage}-{timestamp}-{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        commit = ""
    metadata = {
        "config": cfg.to_dict(),
        "git_commit": commit or None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device_requested": cfg.device,
    }
    (path / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
