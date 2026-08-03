# Research scope

Track A is unconditional, data-free-at-distillation FreeFlow. Track B is a separately trained
arbitrary-state/two-time model for alternating flow jumps with MRI data consistency. Track B may
share backbone blocks but must retain separate wrappers, objectives, configurations, checkpoints,
and result tables.

Initial non-goals are multi-coil sensitivity estimation, 3-D volumes, clinical claims, latent VAEs,
large transformers, and conditional reconstruction distillation.

