# Implementation Plan: Compute-Efficient FreeFlow for Accelerated MRI

## 1. Handoff purpose

You are initializing a research codebase for a personal Master's-level project on data-free flow-map distillation for accelerated MRI reconstruction. You have no prior conversation context; this document is the source of truth.

Your immediate task is to scaffold a clean, tested, configuration-driven repository. Do **not** start a full training run, download the full fastMRI dataset, or claim a paper-faithful FreeFlow reproduction until the mathematical objective has been audited against the paper. Implement small smoke-testable components and leave explicit boundaries for the research-specific objectives.

The available hardware will usually be one of:

- NVIDIA T4, approximately 15 GB VRAM;
- NVIDIA L4, approximately 22.5 GB usable VRAM.

The project must therefore prioritize 2D models, modest image resolution, mixed precision, gradient accumulation, and reproducible small-scale experiments.

## 2. Research objective

The long-term research question is:

> Can a slow MRI flow/diffusion teacher be compressed into a one- or few-step flow-map student without using real MRI images during the distillation stage, while retaining teacher fidelity, MRI measurement consistency, and reconstruction quality?

The term **data-free** applies only to distillation. The teacher may be trained on real fully sampled MRI data. During student distillation, training states should be obtained from Gaussian prior samples propagated through the frozen teacher, rather than from an external static dataset of real images.

The eventual project should study three levels of difficulty:

1. **Track A — unconditional validation:** train a small MRI rectified-flow teacher and distill its generative flow map. Evaluate teacher–student agreement and one/few-step generation.
2. **Track B — physics-guided reconstruction:** use the learned two-time map inside a measurement-guided sampler with explicit k-space data consistency.
3. **Track C — optional research extension:** condition the student flow map on synthetic MRI measurements generated from teacher samples. This is the most novel and risky part, and must not block Tracks A and B.

## 3. Scientific positioning

The project sits at the intersection of:

- progressive diffusion distillation and consistency models;
- flow-map matching and self-distillation;
- FreeFlow-style distillation from the prior without an external dataset;
- score/flow priors for MRI inverse problems;
- one/few-step MRI reconstruction with explicit measurement consistency.

Closest references:

1. Salimans and Ho, *Progressive Distillation for Fast Sampling of Diffusion Models* (2022): https://arxiv.org/abs/2202.00512
2. Song et al., *Consistency Models* (2023): https://arxiv.org/abs/2303.01469
3. Boffi et al., *Flow Map Matching with Stochastic Interpolants* (2024/2025): https://arxiv.org/abs/2406.07507
4. Frans et al., *One Step Diffusion via Shortcut Models* (2024/2025): https://arxiv.org/abs/2410.12557
5. Boffi et al., *How to Build a Consistency Model: Learning Flow Maps via Self-Distillation* (2025): https://arxiv.org/abs/2505.18825
6. Tong et al., *Flow Map Distillation Without Data (FreeFlow)* (2025): https://arxiv.org/abs/2511.19428
7. Jalal et al., *Robust Compressed Sensing MRI with Deep Generative Priors* (2021): https://arxiv.org/abs/2108.01368
8. Chung and Ye, *Score-Based Diffusion Models for Accelerated MRI* (2021/2022): https://arxiv.org/abs/2110.05243
9. Chung et al., *Diffusion Posterior Sampling for General Noisy Inverse Problems* (2022/2023): https://arxiv.org/abs/2209.14687
10. *Fast Controllable Diffusion Models for Undersampled MRI Reconstruction* (PPN, 2023): https://arxiv.org/abs/2311.12078
11. *Highly Undersampled MRI Reconstruction via a Single Posterior Sampling of Diffusion Models* (SSDM-MRI, 2025): https://arxiv.org/abs/2505.08142
12. *Consistency Models as Plug-and-Play Priors for Inverse Problems* (PnP-CM, 2025): https://arxiv.org/abs/2509.22736

Important gap: SSDM-MRI performs one-step conditional MRI reconstruction but uses real MRI-derived training pairs during distillation. PnP-CM reaches roughly 2–4 neural function evaluations but trains/distills its MRI consistency model on real fastMRI images. FreeFlow is data-free during distillation but is not designed or validated for MRI inverse problems. The proposed project explores their intersection.

## 4. Scope and non-goals

### Initial scope

- 2D Cartesian MRI only.
- fastMRI knee, single-coil subset.
- Complex-valued images represented by two real channels.
- Retrospective undersampling with acceleration factors `R=4` and `R=8`.
- Initial resolution: `128x128` for smoke/research debugging, then `192x192`; `256x256` only for a final L4 experiment if the lower-resolution pipeline is stable.
- Convolutional U-Net, approximately 30–60 million parameters at full configuration.
- Rectified-flow teacher with a deterministic probability-flow ODE.
- Student supporting arbitrary source and target times `(s, t)` and 1, 2, or 4 NFE sampling.

### Explicit non-goals for the initial repository

- No 3D volumes.
- No DiT/large Vision Transformer.
- No latent diffusion or separately trained VAE in the first implementation.
- No multi-coil sensitivity estimation.
- No prospective clinical validation.
- No diagnosis, clinical claims, or replacement of radiologist assessment.
- No full FreeFlow/ImageNet reproduction.
- No expensive hyperparameter sweep.

## 5. Core mathematical conventions

Use one time convention consistently throughout code and documentation:

- `t = 0`: Gaussian prior/noise.
- `t = 1`: MRI data distribution.
- Clean image: `x_data`.
- Prior sample: `z ~ N(0, I)` with the same tensor shape as `x_data`.
- Linear interpolation:

  `x_t = (1 - t) * z + t * x_data`

- Rectified-flow target velocity:

  `v_target = x_data - z`

- Teacher objective:

  `L_RF = E[w(t) * ||v_theta(x_t, t) - (x_data - z)||^2]`

The default weighting may initially be `w(t) = 1`. Make the time sampler and weighting configurable.

The teacher sampling ODE is:

`dx/dt = v_theta(x, t)`, integrated from `t=0` to `t=1`.

Implement Euler and Heun solvers. Euler is needed for debugging and reproducibility; Heun should be the default teacher sampler.

Define the student as a two-time map:

`Phi_psi(x_s, s, t, condition=None) -> x_t_hat`

Internally, a numerically convenient residual parameterization is recommended:

`x_t_hat = x_s + (t - s) * u_psi(x_s, s, t, condition)`

This guarantees identity when `s == t`, subject to floating-point error. Unit-test this property.

## 6. MRI forward model and data consistency

For the initial single-coil setting:

`y = A(x) + epsilon = M * FFT2c(x) + epsilon`

where:

- `x` is a complex-valued image represented as two real channels;
- `FFT2c` is an orthonormal centered 2D FFT;
- `M` is a binary Cartesian sampling mask;
- `epsilon` is optional complex Gaussian measurement noise.

Implement paired `fft2c` and `ifft2c` utilities using `torch.fft`, with explicit conversion between two-channel real tensors and native complex tensors. Test round-trip accuracy.

Hard data-consistency projection:

1. Compute `k_hat = FFT2c(x_hat)`.
2. Replace sampled coefficients: `k_dc[M] = y[M]`.
3. Return `x_dc = IFFT2c(k_dc)`.

Also implement a soft consistency update with configurable weight for noisy measurements, but keep hard projection as the initial default for noiseless retrospective experiments.

For reconstruction, do not assume that an unconditional one-step map alone solves the inverse problem. Track B should expose a configurable measurement-guided sampling loop that alternates flow-map jumps and data-consistency operations. Its exact algorithm is experimental and must be benchmarked against the teacher under the same schedule.

## 7. Data handling

### Dataset

Use the official fastMRI knee single-coil data. The repository must never commit MRI files, credentials, download links, or derived patient-identifiable information. Require the user to accept the fastMRI data-use terms independently.

Read the dataset root from either a CLI/config field or the environment variable `FASTMRI_ROOT`. Do not hardcode machine-specific paths.

### Splitting

- Split by volume/patient identifier, never by slice.
- Persist split manifests as small JSON or CSV files containing relative file identifiers and slice indices.
- Support a deterministic subset selector for smoke tests and low-compute experiments.
- Central slices should be selected using configurable fractional slice bounds, for example the central 60–80% of each volume.

### Preprocessing

Recommended initial pipeline:

1. Load single-coil k-space from HDF5.
2. Reconstruct complex image with centered inverse FFT.
3. Complex center-crop to the configured square resolution.
4. Normalize using a scale derived only from available measurements, not from unseen fully sampled target information. A practical initial choice is the 99th percentile magnitude of the zero-filled reconstruction, clamped by a small epsilon.
5. Return the scale so metrics and visualizations can be interpreted consistently.

For unconditional teacher training, measurement-derived normalization is awkward because no undersampling mask is intrinsically required. Make normalization a named, versioned policy. Support at least:

- `train_global_std`: statistics computed only on the training split;
- `zero_filled_p99`: scale derived from a reproducibly generated training mask.

Choose one policy per experiment and record it in the checkpoint metadata. Never mix policies across teacher, student, and evaluation.

### Masks

Implement reproducible 1D Cartesian variable-density masks with a fully sampled low-frequency center. The mask generator must accept:

- acceleration factor;
- center fraction;
- random seed;
- image shape.

Start with fixed canonical masks for evaluation and randomized masks for robustness experiments. Save the exact mask parameters in each result artifact.

## 8. Model design

### Teacher U-Net

Implement a time-conditioned 2D U-Net with:

- input/output channels: 2;
- sinusoidal or Fourier time embedding followed by an MLP;
- residual blocks with GroupNorm and SiLU;
- base channels configurable, default 64 for research configs;
- channel multipliers `[1, 2, 2, 4]`;
- two residual blocks per level;
- attention only at the lowest spatial resolution, and optional/off in smoke tests;
- optional gradient checkpointing;
- no batch normalization;
- EMA weights for evaluation.

Avoid dependencies on a very large framework if a compact native PyTorch implementation is sufficient. If using MONAI or diffusers, isolate the adapter so the rest of the codebase does not depend on library-specific tensor conventions.

### Flow-map student

The student may initially reuse the teacher backbone but must additionally embed both `s` and `t`, or equivalently `s` and `delta_t = t - s`. Keep teacher and student classes distinct even if they share blocks.

Required interfaces:

```python
teacher.velocity(x_t, t, condition=None) -> velocity
teacher.sample(z, schedule, solver="heun") -> x_1
student.map(x_s, s, t, condition=None) -> x_t_hat
student.sample(z, schedule, condition=None, dc_operator=None) -> x_1_hat
```

Do not silently treat a velocity network as a flow-map network. Their training targets and semantics differ.

### Optional conditional extension

Track C may condition on:

- zero-filled complex reconstruction `x_zf`;
- binary mask `M`;
- acceleration factor or other acquisition metadata.

Prefer a simple concatenation/encoder baseline before ControlNet-style modules. The initial conditional signature should exist, but full Track C training need not be implemented in the first scaffold.

## 9. Distillation strategy

Because the public FreeFlow repository currently emphasizes pretrained weights and sampling rather than a complete PyTorch training pipeline, do not invent a “FreeFlow” loss from memory.

Implement distillation behind a strategy interface with at least:

1. `teacher_map_regression`: a transparent baseline that samples `z`, selects `(s,t)`, obtains `x_s` and `x_t` by integrating the frozen teacher, and regresses the student map to `x_t`.
2. `freeflow`: reserved for the paper-faithful objective after auditing the paper equations, including its mechanism for correcting compounding/student rollout errors.

The baseline is still data-free during distillation because it uses Gaussian prior samples and teacher trajectories rather than real MRI images. It is not necessarily equivalent to FreeFlow and must be labeled accurately in logs and reports.

### Online trajectory generation

Default to online teacher trajectory generation:

1. Sample `z ~ N(0,I)`.
2. Integrate the frozen teacher along a configurable grid.
3. Select source/target states from that trajectory.
4. Train the student map.

Provide an optional synthetic trajectory cache only for debugging and throughput experiments. Clearly label cached training as a finite synthetic dataset; it may reintroduce a form of trajectory coverage mismatch and should not be the primary FreeFlow experiment.

### Student rollout correction

Design the API so a later objective can generate off-teacher states from partial student rollouts and ask the teacher for corrective targets. The trainer should be able to choose a state source:

- `teacher_trajectory`;
- `student_rollout`;
- `mixed`.

Do not implement a guessed loss. Add a paper-audit checklist and tests for the selected equations before enabling `method=freeflow`.

## 10. Reconstruction tracks

### Track A: unconditional generation and teacher compression

This is the first scientific milestone.

Evaluate:

- teacher samples at 16 and 32 NFE;
- student samples at 1, 2, and 4 NFE;
- paired teacher–student error using the same prior seeds;
- identity and semigroup consistency of the student map;
- throughput, peak VRAM, and wall-clock latency.

Primary paired metrics can include normalized MSE, PSNR, SSIM on magnitudes, and complex-domain relative error. Distributional metrics should not rely solely on ImageNet Inception features. If no validated MRI encoder is available, report paired teacher agreement and clearly mark distribution metrics as future work.

### Track B: physics-guided reconstruction

Given held-out real measurements:

1. Generate an initial zero-filled reconstruction.
2. Initialize the generative trajectory using a documented measurement-dependent rule.
3. Alternate one/few student map jumps with k-space consistency.
4. Compare against the same procedure using the teacher velocity solver.

Initial baselines:

- zero-filled reconstruction;
- hard data-consistency projection of the initial model output;
- teacher flow with 16/32 NFE and matched guidance;
- distilled student with 1/2/4 NFE;
- optional simple supervised U-Net only if time permits, clearly separated from the data-free distillation claim.

Reconstruction metrics:

- NMSE;
- PSNR on magnitude images;
- SSIM on magnitude images;
- observed k-space residual;
- latency and peak VRAM.

Evaluate `R=4` first, then `R=8`. Use fixed held-out masks and seeds.

### Track C: synthetic-condition distillation

Optional advanced experiment:

1. Sample `z` from the prior.
2. Generate `x_teacher` by integrating the frozen unconditional teacher.
3. Sample a known MRI operator `A` and generate `y = A(x_teacher) + epsilon`.
4. Train a conditional student map `Phi(x_s, s, t | y, A)`.
5. Use no real MRI images during this distillation stage.
6. Evaluate on real held-out fastMRI measurements.

This tests whether physics-generated synthetic conditions are sufficient for conditional map distillation. It is higher risk because teacher-generated anatomy and pathology coverage bounds the student. Treat it as an extension, not the minimum viable thesis result.

## 11. Compute profiles

Create three committed YAML configurations.

### `smoke.yaml`

- resolution: 64 or 128;
- base channels: 32;
- channel multipliers: `[1, 2, 2]`;
- attention: disabled;
- batch size: 2–4;
- at most 1,000 optimizer steps;
- synthetic tensors or at most a handful of local HDF5 examples;
- designed to finish in minutes.

### `t4.yaml`

- target resolution: 192;
- base channels: 64;
- channel multipliers: `[1, 2, 2, 4]`;
- attention only at bottleneck;
- microbatch: 2, with configurable fallback to 1;
- gradient accumulation: 8 or 16;
- FP16 autocast with GradScaler;
- gradient checkpointing enabled;
- teacher sampling grid during distillation: start at 8–16 steps, increase only after profiling;
- dataloader workers: conservative default, e.g. 4.

### `l4.yaml`

- target resolution: 192 initially; optional 256 final run;
- same architecture, microbatch 2–4 after profiling;
- BF16 preferred if the software stack supports it reliably, otherwise FP16;
- gradient accumulation adjusted to target the same effective batch as T4;
- gradient checkpointing configurable rather than mandatory.

Do not encode optimistic throughput assumptions. Add a profiling command that reports:

- trainable parameter count;
- one forward/backward memory peak;
- examples/second;
- teacher trajectory generation time;
- estimated time per 10,000 optimizer steps.

The program should fail with an actionable message on OOM; it should not silently change scientific hyperparameters.

## 12. Suggested repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
│   ├── smoke.yaml
│   ├── t4.yaml
│   └── l4.yaml
├── docs/
│   ├── research_scope.md
│   ├── data_setup.md
│   ├── math_conventions.md
│   ├── experiment_protocol.md
│   └── freeflow_paper_audit.md
├── src/freeflow_mri/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── fastmri.py
│   │   ├── masks.py
│   │   ├── normalization.py
│   │   └── splits.py
│   ├── physics/
│   │   ├── complex.py
│   │   ├── fft.py
│   │   ├── operators.py
│   │   └── data_consistency.py
│   ├── models/
│   │   ├── blocks.py
│   │   ├── unet.py
│   │   ├── teacher.py
│   │   └── flow_map.py
│   ├── flows/
│   │   ├── interpolation.py
│   │   ├── schedules.py
│   │   └── solvers.py
│   ├── distillation/
│   │   ├── base.py
│   │   ├── teacher_map_regression.py
│   │   ├── freeflow.py
│   │   └── rollout.py
│   ├── reconstruction/
│   │   ├── guided_sampler.py
│   │   └── initialization.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── checkpointing.py
│   │   ├── ema.py
│   │   └── reproducibility.py
│   └── evaluation/
│       ├── metrics.py
│       ├── generation.py
│       ├── reconstruction.py
│       └── profiling.py
├── scripts/
│   ├── create_splits.py
│   ├── compute_normalization_stats.py
│   ├── train_teacher.py
│   ├── distill_student.py
│   ├── evaluate_generation.py
│   ├── evaluate_reconstruction.py
│   └── profile_model.py
└── tests/
    ├── test_complex.py
    ├── test_fft.py
    ├── test_masks.py
    ├── test_data_consistency.py
    ├── test_interpolation.py
    ├── test_solvers.py
    ├── test_flow_map.py
    ├── test_dataset_splits.py
    └── test_smoke_train.py
```

The exact tree may be simplified, but preserve separation between MRI physics, teacher velocity modeling, student flow maps, distillation strategies, and reconstruction algorithms.

## 13. Engineering choices

Recommended stack:

- Python 3.11;
- PyTorch 2.x;
- `h5py` for fastMRI files;
- `numpy`, `scipy`, `scikit-image`;
- `PyYAML` or a lightweight typed configuration system;
- TensorBoard as the zero-account default logger, with optional Weights & Biases support;
- `pytest`, `ruff`, and optionally `mypy`;
- `safetensors` where practical for model weights.

Keep dependencies minimal and pin broad compatible ranges rather than exact CUDA-specific builds. Document how to install the correct PyTorch build separately.

Required engineering behavior:

- every run receives a unique output directory;
- save the fully resolved config, Git commit, package versions, seed, and device information;
- checkpoint model, optimizer, scheduler, scaler, EMA, and RNG state;
- support exact resume from checkpoint;
- deterministic validation masks and sample seeds;
- patient-level split assertions;
- no destructive overwrite of previous runs;
- clear distinction between `best` and `last` checkpoints;
- periodic sample grids showing magnitude, phase, error maps, and k-space residuals.

## 14. Testing requirements

Before any real training, the following must pass:

1. Complex conversion round trip preserves values.
2. `ifft2c(fft2c(x))` reconstructs `x` to numerical tolerance.
3. Mask generation is deterministic for a fixed seed and approximately matches the requested acceleration.
4. Hard data consistency produces zero residual on observed noiseless coefficients.
5. Rectified-flow interpolation returns `z` at `t=0` and `x_data` at `t=1`.
6. Euler/Heun solve a simple analytic constant-velocity ODE correctly.
7. Student map is the identity for `s=t` by construction.
8. No volume identifier appears in more than one dataset split.
9. One teacher training step runs on CPU and CUDA if available.
10. One baseline distillation step runs with the teacher frozen and produces finite gradients only for the student.
11. A tiny overfit test can fit a handful of synthetic images.
12. A smoke command completes end to end without requiring the full dataset.

## 15. Metrics and reporting

Log losses and metrics separately for teacher training, student distillation, and reconstruction.

### Teacher

- rectified-flow velocity MSE;
- validation velocity MSE;
- sample norms and finite-value checks;
- generation latency at each NFE.

### Student

- map regression/correction loss;
- paired teacher–student complex relative error;
- magnitude PSNR and SSIM for paired teacher outputs;
- identity error `Phi(x,s,s)-x`;
- approximate semigroup error:

  `||Phi(Phi(x,s,u),u,t) - Phi(x,s,t)||`.

### Reconstruction

- NMSE, PSNR, SSIM;
- normalized observed k-space residual;
- latency per slice;
- peak allocated VRAM;
- NFE count, with teacher and student calls counted explicitly.

Always report mean, standard deviation, and number of volumes/slices. Aggregate by volume where possible so volumes with many slices do not dominate the result.

## 16. Milestones and gates

### Milestone 0 — repository initialization

Deliver:

- package skeleton and installation instructions;
- typed/config-validated smoke, T4, and L4 configs;
- MRI FFT/operator/data-consistency utilities;
- minimal U-Net, teacher, flow-map student, and solver interfaces;
- synthetic smoke dataset;
- tests and CI for CPU smoke checks;
- documentation of all mathematical conventions.

Gate: all unit tests pass and the end-to-end smoke pipeline finishes in minutes.

### Milestone 1 — small teacher at 128x128

Deliver:

- fastMRI single-coil loader and split manifests;
- teacher training and EMA sampling;
- overfit test, then a small real-data run;
- profiling on the available GPU.

Gate: teacher generates structurally plausible samples and training is stable before increasing resolution.

### Milestone 2 — transparent data-free baseline

Deliver:

- online teacher trajectory generation;
- `teacher_map_regression` student;
- 1/2/4 NFE evaluation;
- teacher–student paired comparisons.

Gate: student beats a naive one-step Euler teacher approximation at matched 1 NFE, or the failure is analyzed quantitatively.

### Milestone 3 — paper-faithful FreeFlow objective

Deliver:

- completed `docs/freeflow_paper_audit.md` mapping each implemented term to an equation/algorithm in the paper;
- rollout/error-correction objective;
- regression tests against small deterministic examples;
- ablation against `teacher_map_regression`.

Gate: the method is accurately labeled FreeFlow only after this audit.

### Milestone 4 — 192x192 reconstruction

Deliver:

- measurement-guided teacher and student samplers;
- `R=4` reconstruction benchmark;
- zero-filled, teacher, and 1/2/4 NFE student comparisons;
- compute/quality Pareto plot.

Gate: observed k-space consistency is correct and reconstruction quality is competitive with the teacher at materially lower NFE.

### Milestone 5 — final experiments

Possible additions, in priority order:

1. `R=8`;
2. real-data versus data-free distillation comparison;
3. student-rollout correction ablation;
4. out-of-distribution masks;
5. synthetic-condition Track C;
6. one final 256x256 run on L4.

Do not add multi-coil or 3D unless all earlier gates pass and compute remains available.

## 17. Initial agent deliverables

For the first repository-initialization pass, perform only Milestone 0. In particular:

1. Inspect the existing workspace and preserve any user files.
2. Initialize the package structure without downloading data.
3. Add configuration schemas and the three compute profiles.
4. Implement and test complex tensor utilities, centered FFTs, masks, the single-coil operator, and data consistency.
5. Implement a minimal time-conditioned U-Net and explicit teacher/student wrappers.
6. Implement rectified-flow interpolation plus Euler and Heun solvers.
7. Add a synthetic dataset and tiny CPU smoke training loops for teacher and baseline map regression.
8. Add clear `NotImplementedError` or disabled config handling for unaudited FreeFlow-specific loss terms; do not leave misleading silent stubs.
9. Add README commands for install, test, profile, smoke teacher training, and smoke distillation.
10. Run formatting, linting, and tests and report their results.

Do not launch long jobs. Do not fabricate performance numbers. Document assumptions and unresolved decisions.

## 18. Main risks

1. **FreeFlow reproduction risk:** the public implementation may not expose the full training code. Mitigation: build a transparent baseline first and audit every paper-specific term.
2. **Teacher quality ceiling:** the student cannot recover anatomical or pathological modes absent from the teacher.
3. **MRI hallucination risk:** perceptual quality alone is insufficient. Always report k-space consistency and distortion metrics.
4. **Compute multiplication:** online distillation repeatedly evaluates the teacher. Profile trajectory generation before scaling.
5. **Normalization leakage:** target-derived per-image normalization can leak unavailable information. Version and audit normalization policies.
6. **Slice leakage:** random slice splits can place the same patient in train and test. Enforce volume-level splits.
7. **Unfair NFE comparison:** data-consistency operations and teacher calls must be counted and timed consistently.
8. **Resolution creep:** prove the method at 128/192 before 256; avoid 320 and 3D in the initial project.

## 19. Definition of project success

Minimum successful result:

- a reproducible 2D complex-valued MRI rectified-flow teacher;
- a student distilled without real MRI data during the student stage;
- evidence that the student approximates the teacher at 1–4 NFE;
- a physics-guided reconstruction experiment on held-out fastMRI data;
- explicit quality-versus-compute comparisons and honest failure analysis.

Strong result:

- FreeFlow-style rollout correction improves over simple teacher-map regression;
- the student approaches teacher reconstruction metrics with materially lower latency/NFE;
- data-free distillation matches or exceeds real-image-based distillation under the same compute budget;
- performance remains stable under unseen masks or acceleration factors.

The repository should favor scientific clarity, reproducibility, and low-compute iteration over architectural scale.
