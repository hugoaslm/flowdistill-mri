# FreeFlow paper audit

Status: **Phase 0 equation-level reproduction; large-scale numerical equivalence is not claimed.**

| Paper concept | Local implementation | Audit guard |
| --- | --- | --- |
| Prior-anchored map `f(z,delta)=z+delta F(z,delta)` | `PriorAnchoredFreeFlow` | identity at `delta=0` |
| Prediction identity (paper Eq. 7-9) | `prediction_loss` | analytic teacher and stop-gradient tests |
| Finite-difference duration derivative | adjacent `delta,delta+h` evaluations | deterministic interval test |
| Endpoint re-noising | `correction_losses` | explicit `teacher_time=1-r` test |
| Auxiliary noising velocity | separate corrector U-Net | student/teacher gradient isolation |
| Correction gradient (paper Eq. 11) | terminal velocity dotted with stopped discrepancy | gradient isolation |
| Prediction/correction balance | adaptive weight plus delay/warmup | schedule unit tests |

The implementation was informed by a working local reproduction in `rs-flow-vqa`, but all
remote-sensing conditioning, variable token masking, and its reversed time convention were removed.

Before scientific FreeFlow claims, audit the final paper appendix against this file, add JVP parity
tests, record weighting/clipping choices, and ablate prediction-only, correction-only, and combined
training.

