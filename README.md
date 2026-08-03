# FlowDistill MRI

CPU-first research code for data-free flow-map distillation and accelerated single-coil MRI.

The project deliberately separates:

- **Track A:** a prior-anchored reproduction of FreeFlow for unconditional MRI generation;
- **Track B:** a future arbitrary-state, two-time flow map for data-consistent MRI reconstruction.

Track B is FreeFlow-inspired and is not labeled as a paper-faithful FreeFlow reproduction.

## Install and test

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
ruff check .
pytest
```

PyTorch installation may be selected separately for CPU or the CUDA runtime. Google Colab can use
its preinstalled PyTorch and then run `pip install -e . --no-deps`.

## Commands

```bash
flowdistill-mri inspect --config configs/smoke_ci.yaml
flowdistill-mri train-teacher --config configs/smoke_ci.yaml --output outputs/demo/teacher
flowdistill-mri distill-freeflow --config configs/smoke_ci.yaml \
  --teacher-checkpoint outputs/demo/teacher --output outputs/demo/freeflow
flowdistill-mri evaluate-generation --config configs/smoke_ci.yaml \
  --teacher-checkpoint outputs/demo/teacher --student-checkpoint outputs/demo/freeflow
flowdistill-mri smoke --tier ci       # intentionally not run during repository initialization
flowdistill-mri smoke --tier local    # intentionally not run during repository initialization
```

Pass `--output <checkpoint-dir> --resume` to teacher or distillation training to continue an exact
compatible checkpoint. Non-empty output directories are never overwritten without `--resume`.

Every run writes its resolved configuration and runtime metadata into a unique run directory unless
an explicit resume checkpoint is supplied. MRI data and checkpoints are ignored by Git.

## Colab

Open [`notebooks/flowdistill_mri_colab.ipynb`](notebooks/flowdistill_mri_colab.ipynb) in Colab. The
notebook clones this repository, optionally mounts Google Drive, and calls the same CLI used above.
Long-running cells are opt-in.

## Data

No dataset is downloaded automatically. Future fastMRI support requires independent acceptance of
the fastMRI data-use agreement and a configured `FASTMRI_ROOT`. See `docs/data_setup.md`.

## Scientific status

The prediction/correction implementation is mapped to the FreeFlow equations in
`docs/freeflow_paper_audit.md`. The public FreeFlow PyTorch repository provides inference code and
ported weights rather than its original JAX training pipeline, so this project keeps equation-level
tests and does not claim numerical equivalence to the authors' training system.

No software license has been selected yet.

The approved phased roadmap is recorded in `docs/implementation_plan.md`; the original research
brief remains at the repository root for provenance.
