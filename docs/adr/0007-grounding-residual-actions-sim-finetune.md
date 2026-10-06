# ADR 0007 — Grounded LeWM, joint-residual actions, and sim fine-tuning

- Status: proposed (results pending; pipeline `scripts/slurm/lewm_pipeline.sh`)
- Date: 2026-10-07
- Follows: `docs/results/protocol-v2-baseline-vs-lewm.md`, ADR 0004, ADR 0006

## Context

The protocol-v2 results pointed to two problems with `lewm-depth` that are
about the model rather than the planner:

1. **The latent drops state it is given.** The 33-d proprio input is
   exactly the EV5 probe targets minus base position, yet the linear probe
   recovers ang_vel at R² 0.11, gravity at 0.51, and joint_pos at 0.80. The
   prediction loss rewards discarding hard-to-predict inputs, and SIGReg
   does not push back.
2. **The predictor under-reacts to actions.** EV2 ASR is 0.3–0.7 (ideal 1).
   Actions are absolute joint-position commands. Under a tracking
   controller they are close to the current joint positions, so the history
   predicts most of them. The informative part, the command minus where the
   joint is, is small next to each command's per-joint standardization.
   A δ = 0.02 rad perturbation moves the normalized input by only a few
   hundredths.

## Decisions

### Arm 1 — grounded (`configs/jepa-wm/lewm_depth_grounded.yaml`)

This is the E1.3 full-PSG arm: `loss_weights.state = transition = 1`,
everything else as `lewm_depth.yaml`. ADR 0004 left it off as "close to an
autoencoding target", but item 1 shows that autoencoding the proprio is
exactly what is missing. Expected effect: EV5 probe R² rises on the proprio
components, and ε_floor drops with it.

### Arm 2 — joint-residual action input (`lewm_depth_residual.yaml`)

`model.action_input: joint_residual`. The predictor is conditioned on

    r[t, f, j] = (command[t, f, j] − q̂[t, j] − μ_r[f, j]) / σ_r[f, j]

Here f indexes the tick's 10 control frames, and q̂[t] is the joint
position the **state head decodes from latent t** (un-normalized). μ_r and
σ_r are training-split statistics of `command(t + f/50 Hz) − joint_pos(t)`
per (frame, joint) (`command_residual_stats`, stored as
`normalization.residual_mean/std`).

- **Why decoded and not measured joints.** For predicted frames, measured
  joints are future observations. Feeding them through the action input in
  an open-loop rollout would leak the future into EV5 / EV1-sim / EV2 and
  into planning. The state head is defined on every latent, so the same
  rule applies to encoded and predicted frames, in training and in every
  eval. That is why joint_residual requires `loss_weights.state > 0`.
- **The conversion lives inside the model.** Datasets, the planner, and
  every eval still pass normalized absolute commands, so no eval code
  branches on the arm. The normalization the model needs is held in
  buffers, set by training via `set_input_normalization`, and restored
  from the state dict.
- **No gradient through q̂.** The state head is trained by `state_loss`
  alone, so q̂ keeps meaning "joint positions". The prediction loss cannot
  repurpose it.
- With joint_residual, the state head is part of inference, so EV7 counts
  it as inference parameters (`training_only_modules`).

### Arm 3 — sim fine-tune (`lewm_depth_residual_simft.yaml`)

This arm starts from the same seed's arm-2 `last.pt` (weights only, fresh
optimizer, parent's normalization statistics). It trains 10 epochs at
lr 1e-4 on GrandTour windows mixed with MuJoCo windows
(`configs/sim/finetune_episodes.yaml`, `sim_repeat: 2`, about 1/3 of
windows).

- The episodes are the scripted trot plus i.i.d. per-frame noise (0.03 rad,
  as in EV1-sim) plus a **per-tick, per-joint held residual** (0.1 rad).
  That residual has the shape and scale of the EV3 planner's search space,
  so the model sees what such actions do. GrandTour's logged actions cannot
  provide this counterfactual variation.
- Seeds start at 1000, so no fine-tuning terrain or episode equals an eval's
  (EV1-sim 0–7/100–103, EV2 200–203, EV3/4/6 0–1). Terrain *kinds and
  levels* overlap with EV3 by design.
- **Sim evals of this arm are not zero-shot.** Δ_s2r for it measures
  something different from the other arms, and EV3/EV4/EV6 differences
  partly reflect in-domain training. Compare it with arm 2 to isolate the
  fine-tune, and check its real EV5 for forgetting.
- Episodes are rendered by `quadwm sim-collect` (depth only) before
  training. Training refuses to render (`episode_set(collect=False)`), so
  DDP ranks never wait on rank 0.

### Seeds and pipeline

Each arm runs seeds 4551–4553 (protocol §11 needs ≥ 3). `QUADWM_SEED`
selects the seed and the run directory (`<name>-s<seed>`). The mission split
stays fixed by `data.split_seed`. `scripts/slurm/lewm_pipeline.sh` submits:

- sim-collect;
- every train job, each followed by a real eval and a 3-shard sim eval
  (`afterok`, `--kill-on-invalid-dep`);
- arm 3 after its arm-2 parent;
- a final `quadwm report` (`afterany`).

Its jobs set `GPU_SELECT=slurm` (new in `jepa_baseline.sbatch` and
`jepa_eval.sbatch`; the default stays `freest`), as the sim eval has done
since `8efbd77`. Jobs that start together otherwise race to the same
"freest" card.

## Consequences

- `normalization_stats` now also writes `residual_mean/std` when given
  `control_hz`, which training always passes. Existing keys are unchanged.
  Absolute-input models ignore the new keys.
- Old checkpoints load unchanged (`action_input` defaults to `absolute`).
- No protocol config changed. EV5/EV1-sim/EV2/EV3 numbers for these arms
  are comparable to `protocol-v2-baseline-vs-lewm.md`, with the arm-3
  zero-shot caveat above.
- Not done here: the baseline arms (its proprio is squeezed to 16 dims, and
  the same residual input would apply), the planner cost/uncertainty
  fixes, and the ε − floor / Δs metric changes. Those are separate
  decisions.
