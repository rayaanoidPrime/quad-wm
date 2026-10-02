# ADR 0005 — Shared-protocol evaluation on GrandTour (`quadwm eval`)

- Status: accepted
- Date: 2026-10-01

## Context

Training only ran the E1.1 check (per-step latent MSE against persistence,
in each model's own latent space, on 8 batches). Those numbers cannot be
compared across models: protocol §9 excludes latent-space and training-loss
metrics. Protocol §1–2 compares models in *probe space* instead. Both Track 1
recipes (frozen V-JEPA baseline, from-scratch LeWM) are trained on GrandTour
only, and no walking controller exists in simulation yet (ADR 0003).

## Decision

`quadwm eval --config <training config> [--checkpoint] [--eval-config]`
scores one checkpoint using `configs/eval/protocol.yaml`, which is shared by
every model. The output JSON goes to `<run>/eval/`. `quadwm report`
aggregates these JSONs across seeds.

What runs:

| Protocol item | Implementation |
|---|---|
| §1 probes | Linear and MLP (2×256, GELU) heads on frozen latents. Same AdamW, LR, cosine schedule, 4000-step budget and seed for every model. Inputs and targets are standardized with probe-split statistics. No early stopping, because selecting a stopping point would need a fourth split. |
| §1.3 rule 4 | Probes are fit on the run's `splits.json` probe missions (ADR 0001) and scored on its eval missions. A run whose splits predate ADR 0001 is refused. |
| E1.2 | R² and Pearson r per state component, on encoded latents of the eval missions. |
| EV5 (EV1 metric, real data) | ε_k at k ∈ {1, 5, 12, 25, 50}: mean σ-normalized L2 between the probe applied to the open-loop rollout and the logged state. σ is the per-component std over the eval set. Reported for all 40 dims and per component. |
| EV7 | Parameters (inference, training-only heads, frozen encoder), open-loop rollout FPS, batch-1 single-step latency (encode the new frame + one predictor step), and approximate training GPU-hours from `metrics.jsonl`. Rollout FPS times `model.rollout` from a full `context_steps` context, the same recurrence the eval scores. Eval JSONs written before this was fixed timed the baseline from a shorter `rollout_context` window, so their baseline FPS is not comparable. |
| §11 | Mean ± std with n, Mann-Whitney U per horizon, Holm correction across the comparison table. Only reported for n ≥ 3. |

Choices the protocol does not settle:

- **Probe latent for the baseline.** Each frame has 576 tokens × (768
  visual + 16 proprio). The probe latent is the token mean of the
  per-slice-LayerNormed tokens (784-d), the space the predictor is trained
  and scored in. LeWM's latent is its 192-d CLS projection. The probes are
  architecturally identical; only the input width follows the latent size,
  which is unavoidable across models. The parameter count of each probe is
  recorded in the eval JSON.
- **Shared windows across cameras (rule 5).** The baseline sees the RGB
  camera and LeWM the depth camera, so their native windows have different
  timestamps and validity. Rollout starts sit on a fixed 5 s grid per
  mission. A grid point is kept only if every camera in
  `windows.match_observations` has a valid window within 0.1 s, and the
  kept points are thinned evenly to 64 per mission. Each window has 7
  context frames (≥ every model's `context_steps`) and 50 future frames, so
  every model predicts the same frames from the same t. A model with a
  shorter context uses the trailing frames.
- **Base position.** It is re-anchored at the rollout start t (not the
  window's first frame, as in ADR 0002), so `base_pos` at t+k is the
  displacement over the horizon. Displacement cannot be identified from a
  single latent in any case. ε_k is reported over all 40 dims as the
  protocol defines it, and additionally as `all_excl_base_pos`, a clearly
  labeled extra rather than a replacement.
- **References in every curve.** `persistence` is the probe of z_t scored
  against s_{t+k}, which the rollout must beat. `encoded_floor` is the probe
  of the true encoded z_{t+k}, the error left even if the prediction were
  exact.
- **Gait cycle.** k = 12 is kept, because the protocol's horizon set may
  not change silently. The eval measures the gait period from
  foot-contact autocorrelation on moving segments and reports it in
  seconds and ticks. Changing the horizons is then a versioned edit to
  `configs/eval/protocol.yaml` (new `name`) with an ADR.
- **Token cache.** The eval reuses the training cache. A rebuild now takes
  the union of cached and requested frame ids, so eval and training windows
  never evict each other.

## Simulated evals

EV1-sim, Δ_s2r, EV2, EV3, EV4 and EV6 run in `quadwm sim-eval`
(docs/adr/0006). `quadwm report` pairs a checkpoint's real and sim JSONs.

## Consequences

- `configs/eval/protocol.yaml` is part of the protocol. Editing it means
  bumping `name`, and `quadwm report` flags runs evaluated under different
  names, horizons, windows or splits.
- Both camera topics must be materialized under `GRANDTOUR_ROOT` to
  evaluate either recipe under the default config.
- `quadwm eval` results are real-data EV5 only. EV1 (sim) comes from
  `quadwm sim-eval`; the two must not be presented as each other.
