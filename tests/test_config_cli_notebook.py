import json
from pathlib import Path

import pytest

from flowdistill_mri.cli import build_parser
from flowdistill_mri.config import load_config
from flowdistill_mri.reconstruction.two_time import train_two_time_flow_map


def test_configs_and_cli_parse():
    cfg = load_config("configs/smoke_ci.yaml")
    assert cfg.data.kind == "synthetic"
    for path in Path("configs").glob("*.yaml"):
        assert load_config(path).experiment
    args = build_parser().parse_args(["inspect", "--config", "configs/smoke_ci.yaml"])
    assert args.command == "inspect"


def test_notebook_is_valid_and_long_jobs_are_opt_in():
    notebook = json.loads(Path("notebooks/flowdistill_mri_colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert "RUN_CPU_SMOKE = False" in source
    assert "RUN_TEACHER = False" in source
    assert "RUN_DISTILLATION = False" in source
    assert "pip', 'install', '--quiet', 'pyyaml>=6', 'safetensors>=0.4'" in source
    assert "'pull', '--ff-only'" in source


def test_fastmri_download_notebook_is_safe_and_opt_in():
    path = Path("notebooks/download_fastmri_to_drive.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert "RUN_DOWNLOADS = False" in source
    assert "RUN_EXTRACTION = False" in source
    assert "FASTMRI_KNEE_VAL_URL" in source
    assert "FASTMRI_KNEE_TRAIN_URL" in source
    assert "AWSAccessKeyId=" not in source
    assert "CONFIRM_ARCHIVE_DELETION = ''" in source


def test_track_b_is_explicitly_disabled():
    with pytest.raises(NotImplementedError, match="Track B"):
        train_two_time_flow_map()
