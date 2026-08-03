# Implementation roadmap

## Phase 0 — repository foundation

Deliver the package, MRI numerical operators, compact rectified-flow teacher, prior-anchored
FreeFlow student, prediction/correction objectives, deterministic synthetic phantoms, checkpointing,
CLI, two opt-in CPU smoke tiers, tests, CI, and a Colab-first notebook. Track B exposes only an
explicit disabled interface. Repository initialization runs tests but does not run smoke training.

## Phase 1 — CPU validation gate

Run `smoke_ci`, then `smoke_local`. Require deterministic completion, finite losses, correct gradient
boundaries, checkpoint resume, artifacts, and paired teacher/student metrics before GPU use.

## Phase 2 — real-data teacher

Add fastMRI knee single-coil loading, volume-level manifests, normalization policies, and fixed
evaluation masks. Overfit a tiny subset at 64–128 resolution on Colab, profile T4/L4 behavior, and
scale to 192 only after stability.

## Phase 3 — Track A FreeFlow validation

Distill from Gaussian prior samples and frozen-teacher evaluations only. Complete the paper audit,
add JVP/finite-difference parity, and compare prediction-only, correction-only, combined FreeFlow,
and teacher-trajectory regression. Report one-step fidelity, complex and magnitude metrics, latency,
memory, and trajectory error.

## Phase 4 — Track B reconstruction

Implement a separate arbitrary-state `Phi(x_s,s,t)` wrapper, objective, configuration, checkpoint,
and results namespace. Train on teacher-generated transitions with synthetic MRI operators, then
alternate map jumps with data consistency. Label this track FreeFlow-inspired, not paper-faithful.

## Phase 5 — final experiments

Evaluate `R=4` before `R=8`, unseen masks, correction ablations, data-free versus real-data
distillation, and quality/compute Pareto curves. Conditional distillation and 256×256 are optional;
multi-coil and 3-D remain out of scope.
