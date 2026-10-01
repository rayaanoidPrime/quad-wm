# ADR 0006 — Simulated protocol evals (`quadwm sim-eval`)

- Status: accepted (stand-in robot and controller; see Limits)
- Date: 2026-10-01

## Context

ADR 0005 covered what real GrandTour logs support. EV1-sim, Δ_s2r, EV2, EV3,
EV4 and EV6 need a simulator, and ADR 0003 left two gaps open: there was no
walking controller, and there were no models trained on simulator
observations. Both Track 1 recipes are trained on GrandTour only, so every
simulated result here is zero-shot sim transfer.

## Decision

`quadwm sim-eval --config <training config>` runs the simulated half of
the protocol for one checkpoint. Its settings come from
`configs/eval/sim.yaml`, which is shared by every model. That config points
at `configs/eval/protocol.yaml`, so EV1-sim uses exactly the real-data probe
procedure and horizons. `quadwm sim-collect` pre-renders the episodes on a
CPU node. `quadwm report` adds the sim tables and pairs each checkpoint's
real and sim JSONs to give Δ_s2r(k).

### Assets

The `sim` extra installs `robot_descriptions`. When the configured MJCF is
missing, it clones MuJoCo Menagerie (about 2 GB, once) into
`$QUADWM_SIM_ASSETS/mujoco_menagerie` at that package's pinned commit, so
every machine simulates the same robot files.

### Controller: a scripted trot (`sim/controller.py`)

This is the stand-in for the "fixed pretrained ANYmal D controller" that the
recipes assume:

- Diagonal pairs move half a cycle apart at 1.5 Hz. The hip sweeps the foot
  fore and aft, and the knee folds during swing.
- A heading loop shortens the strides on one side to steer.
- Stride length is the speed knob, calibrated so the commanded speed
  roughly matches the speed achieved on flat ground.
- Its output is a GrandTour action: 10 × 12 joint targets per 5 Hz tick.
- The PD gains are raised from Menagerie's kp 100 / kd 2 to kp 300 / kd 8.
  At the lower gains the base sagged to about 0.40 m and the gait barely
  moved. At the higher gains it walks at roughly 0.1–0.6 m/s on flat, rough,
  10° slope and 10 cm stairs without falling.

### What runs

| Item | Implementation |
|---|---|
| EV1-sim | Episodes are 24 s. Each uses a random speed command, one commanded turn, and action noise of σ = 0.03 rad. Probe-fit seeds and eval seeds are disjoint, across flat, rough, slope and stairs terrain. Episodes are rendered once and cached as `.npz` under a key built from the config hash, so every model is scored on identical frames (rule 5). Windows that touch a fall are dropped. The real eval's latent collection and probe scoring (`evaluation/common.py`) are reused unchanged. |
| Δ_s2r(k) | ε_k(real) − ε_k(sim) for the same checkpoint, computed by `quadwm report`. |
| EV2 | Warm up with the controller, save the sim state, and record the nominal continuation A. Then restore the state and replay A + δ·N(0, 1) for δ ∈ {0.02, 0.05, 0.1} rad. Both the true and the model divergence are σ-normalized L2 in probe space. The report gives the ratio of means (the headline) and the median of the per-sample ratios. |
| EV3 | CEM: 300 candidates, 30 elites, 10 iterations, H = 6, replanning every tick and warm-started by shifting the previous plan. It runs on the six terrain tiers. A controller-only arm uses the same seeds as a reference. Metrics follow protocol §4.3: tracking error and fall rate (tiers 1–2), success rate (3–4), and the largest level that at least half the seeds traverse (5–6). |
| EV4 | The EV3 episode on rough terrain under the §5 grid plus the two compound conditions. Retention is forward progress relative to nominal, for both arms. |
| EV6 | A new `unilateral_steps` terrain puts 10 cm blocks under the left feet only, and `mirrored` flips it. Reports the mirrored/original ratio of success and of progress. |

### Departures from the protocol

- **CEM search space.** Per-tick 12-d residuals on top of the controller
  give 72 dims, instead of the raw 720-dim action sequence. The baseline
  recipe already flags 720 dims as beyond anything the CEM references
  validated.
- **CEM cost.** Locomotion has no goal image, so the score is not L2 to a
  goal latent. Instead the frozen sim MLP probe, applied to the predicted
  latents, scores velocity-tracking error plus tilt plus yaw rate, with a
  penalty on residual size. The probe is fit identically for every model,
  so the cost is the same for all of them.
- **EV6 without training.** The protocol trains policies on one-sided
  terrain, then tests on the mirror. With a planner and nothing trained in
  sim, the ratio instead measures whether the world model plus probe
  respond symmetrically to the terrain.
- **EV4 performance** is forward progress over a fixed episode, chosen so
  that retention is a well-defined ratio.

## Limits (state these with any number)

- The robot is ANYmal C (Menagerie) with a pinhole 128² camera at a
  placeholder mount, not ANYmal D with the Boxi fisheye camera (ADR 0003).
  Δ_s2r mixes the sim-to-real gap with this mismatch.
- The models never saw simulator data. Poor EV3 planning may reflect domain
  gap, not model quality. The controller-only arm separates "the planner
  hurts" from "the robot can't do this tier".
- The scripted trot walks rather than runs, and gap and step tiers are hard
  for it. Expect the max-level tiers to sit near the lowest level for both
  arms.
- EV3's learned-policy variant (actor-critic in imagination) is not
  implemented, and the JSON lists it under `not_run`.
- Cost. The planning arm runs about 9k plans of 10 CEM iterations each. For
  LeWM that is minutes to an hour on a GPU. For the V-JEPA baseline every
  candidate is a 576-token × 3-frame predictor rollout, which can take on
  the order of a day. Use `--only` to split the work across jobs; budgets
  must not differ between models.
