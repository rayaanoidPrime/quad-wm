# ADR 0002 — Input standardization and the 40-d probe state

- Status: accepted
- Date: 2026-09-27

## Input standardization

Baseline recipe §2.3 requires per-dimension standardization of actions and
proprioception with training-set statistics; the code previously fed raw
values. `normalization_stats` computes mean/std over **every logged sample
of the training missions** (not only sampled windows), actions per joint
and tiled across the stacked frames. Statistics are written to
`normalization.json` and stored in the checkpoint's config under
`data.normalization`; eval datasets reuse them. Controlled by
`data.normalize` (default true).

Consequence: checkpoints trained before this change expect raw inputs and
are not comparable with new runs.

## Probe state vector

Datasets now emit `state` [T, 40] in the shared-protocol layout
(`STATE_LAYOUT`): base position, base linear/angular velocity, projected
gravity, joint positions, joint velocities, foot contacts. It is never
normalized by the dataset: the protocol normalizes by evaluation-set std
at scoring time.

Base position is **relative to the window's first tick, rotated into that
tick's yaw-aligned frame**. Absolute odometry position is not identifiable
from egocentric input and would make ε_k depend on where a mission started.
