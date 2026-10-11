# ADR 0008 — LeWM-only planning variant: protocol budget, CEM guards, state-head readout

- Status: proposed (results pending)
- Date: 2026-10-11
- Follows: ADR 0006 (and its 2026-10-06 addendum), ADR 0007

## Context

`sim-protocol-v2` plans with CEM at 32 samples / 4 elites / 4 iterations over
72 residual dims (H = 6 × 12 joints). The ADR 0006 addendum cut it from the
protocol's 300/30/10 because the V-JEPA baseline could not finish at that
budget. Reviewing the planner against other implementations (V-JEPA 2-AC,
LeWM, DINO-WM, TD-MPC2, Puppeteer, a Dreamer-MPPI quadruped controller) showed
three weaknesses that are about the planner, not the world model:

1. **The search is too thin.** With 4 elites in 72 dims, the refit std is
   mostly noise and the elite mean is close to arbitrary. Every reference
   uses 100–512 samples and 10–64 elites.
2. **The applied plan is never scored.** CEM returns the final elite mean
   and `run_episode` applies its first tick, but that mean was never rolled
   through the model. The zero residual (the controller alone) is also
   never guaranteed a score. TD-MPC2 and Dreamer-MPPI always keep their
   prior's trajectories in the population, and DINO-WM puts the mean in
   slot 0.
3. **The scorer is a probe fit afterwards.** The frozen MLP probe is fit on
   *encoded* latents from offset-free trot episodes, then applied to
   *predicted* latents from offset actions. Approach-B planners score with
   heads trained jointly with the dynamics. The grounded LeWM arms (ADR 0007,
   `loss_weights.state = 1`) already train such a head: `state_head` decodes
   the 33-d proprio (lin_vel, ang_vel, gravity, joints), which covers every
   cost term.

The V-JEPA baseline has no state head, and the shared protocol needs one
budget for every model, so none of this can go into `sim-protocol-v2`.

## Decisions

- **`configs/eval/sim_lewm.yaml` (`name: sim-lewm-v1`)** is a separate eval
  config, not a protocol version. `model_types: [lewm]` makes `sim-eval`
  refuse other checkpoints. Everything except `name`, `model_types` and
  `planning` is identical to `configs/eval/sim.yaml`, and a test checks
  this. Episodes (and the episode cache), probes, seeds, terrains and the
  controller arm therefore match the protocol runs of the same checkpoint.
- **Budget:** 300 / 30 / 10, the protocol §4.2 numbers. H, init_std,
  clip (0.3 rad), cost weights and residual weight are unchanged from v2,
  so v2 → v1 on the same checkpoint isolates the search.
- **CEM guards** (`cem()` options, all off by default so v2 is
  bit-identical, with the same noise draws):
  - `keep_mean`, `keep_zero`: the current mean and the zero residual
    replace the first two samples of every iteration.
  - `return_best`: apply the lowest-cost candidate scored in any iteration.
    With `keep_zero`, the planner deviates from the controller only when the
    model scores the deviation as cheaper.
  - `std_momentum: 0.5`: a moderate value. With 30 elites the refit is
    already stable, and the momentum only guards against early collapse.
    V-JEPA 2-AC (0.95) and Dreamer-MPPI (β = 0.95) use more, but over
    10–100× more iterations.
- **Two planning arms on identical episodes** via `planning.readouts:
  [probe, state_head]`:
  - `planning`: frozen probe, as in v2.
  - `planning_state_head`: `model.state_head(ẑ)`, un-normalized with the
    run's `proprio_mean/std` and placed in the 40-d state layout. Base
    position and contacts are NaN, since the head does not predict them.

  Both arms share the controller arm and the seeds, so one report compares
  them directly. Configs without `readouts` behave as before (`["probe"]`).
- **The report reads arm names from the results** instead of a fixed
  `("planning", "controller")`. A group whose seeds report different arms
  is an error.

## How to read the results

Use these comparisons, each on the same seeds:

| comparison | isolates |
|---|---|
| residual @ v2 → simft @ v2 | the sim fine-tune (ADR 0007) |
| simft @ v2 → simft @ sim-lewm-v1, `planning` | search budget + guards |
| `planning` → `planning_state_head` (same file) | scorer: post-hoc probe vs jointly trained head |
| residual @ sim-lewm-v1 (optional) | whether a stronger search exploits an off-distribution model |

A stronger optimizer finds model errors more reliably. If the planning arms
fall more often than the controller at v1 but not at v2, the model is
being exploited. That is not evidence that the search is broken.

## Consequences

- `sim-protocol-v2` behaviour, JSON layout and report tables are unchanged.
  New JSONs carry a `planning` block (arm names + planning config).
- sim-lewm-v1 numbers never share a table with sim-protocol-v2. `quadwm
  report` already flags mixed `eval_config`s, and `jepa_report.sbatch`
  takes `SIM_NAME=sim-lewm-v1` to collect them separately.
- Not addressed: no value beyond the 1.2 s horizon, the open-loop scripted
  trot (no 50 Hz balance feedback), and the action-space mismatch for the
  non-fine-tuned arms. The trot remains the sim's low-level controller
  until a learned locomotion policy replaces it.
