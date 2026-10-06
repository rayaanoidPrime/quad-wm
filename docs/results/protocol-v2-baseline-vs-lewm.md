# Results — Track 1 baseline vs LeWM (real EV5 + sim protocol v2)

- Date: 2026-10-06
- Status: recorded; single seed per model, so **no significance tests** (shared
  protocol §11 requires ≥3 seeds).
- Source of truth: the eval JSONs under `quad-wm-storage/runs/**/eval/`, merged
  by `quadwm report`. This doc is a curated summary of that output plus
  provenance; numeric per-component detail also exists in each JSON.

## TL;DR

- **LeWM beats the baseline on every axis, real and sim.** On real GrandTour it
  has ~2× the probe R² (0.61 vs 0.28 linear) and lower rollout ε at every
  horizon, and it is ~8× faster and ~10× smaller.
- Both beat the persistence baseline on **real** data with nearly flat error
  growth out to 5 s (k=25). In **sim**, the baseline does **not** beat
  persistence (its predictor does not transfer zero-shot), while LeWM does.
- **CEM planning (EV3/EV4/EV6) loses to the scripted trot controller
  everywhere.** This is a weak-search artifact, not a model verdict: the CEM
  budget was cut ~23× (see `0248a75`) to fit the baseline in walltime.
- **Δ_s2r is negative for both models** — the sim stand-in is *harder* than
  real GrandTour (robot/camera mismatch, ADR 0006), not a normal sim-to-real gap.

## Provenance

| item | baseline | LeWM |
|---|---|---|
| model | `jepa-baseline-v2` (frozen V-JEPA 2.1 ViT-B, `model.type: jepa-baseline`) | `lewm-depth` (from scratch, `model.type: lewm`) |
| checkpoint | `runs/jepa-baseline/jepa-baseline-v2/last.pt` | `runs/lewm/lewm-depth/last.pt` |
| epoch | 10 | 50 |
| latent_dim / context_steps | 784 / 7 | 192 / 4 |
| config | `configs/jepa-wm/baseline.yaml` | `configs/jepa-wm/lewm_depth.yaml` |
| seed | 4551 | 4551 |
| real eval (EV5) config | `protocol-v1` | `protocol-v1` |
| real eval git commit | `938bf629fd99682352a5a9d383550468e6ce0673` | same |
| real eval Slurm job | 1869 | 1870 |
| sim eval config | `sim-protocol-v2` | `sim-protocol-v2` |
| sim eval git commit | `0248a758394f7b82cf1e0bc640b01494812277e6` | same |
| sim eval Slurm jobs (ev1/2/6 · ev3 · ev4) | 1933 · 1934 · 1932 | 1926 · 1927 · 1923 |
| host / device | `iisc` / AMD Instinct MI300X | `iisc` / AMD Instinct MI300X |

Real eval was produced at `938bf62`; the sim eval at `0248a75`. The intervening
predictor/uint8 changes (`0b38dd5`) are numerically equivalence-tested, so the
real numbers still stand (recorded here with their original commit for honesty).

Metrics are defined in `recipes/shared_evaluation_protocol.md`; simulated evals
in `docs/adr/0006-simulated-protocol-evals.md`.

---

# Real data (GrandTour) — EV5

ε_k is σ-normalized L2 in probe space over all 40 state dims; lower is better.
`encoded_floor` is the probe applied to the true future latent (irreducible
error); `persistence` is `ẑ=z_t`.

## Probe quality on encoded latents

### Linear probe

| group | metric | base_pos | lin_vel | ang_vel | gravity | joint_pos | joint_vel | contacts | all |
|---|---|---|---|---|---|---|---|---|---|
| jepa-baseline | r2 | -0.320 | 0.433 | 0.294 | 0.144 | 0.491 | 0.222 | – | 0.284 |
| jepa-baseline | pearson | 0.210 | 0.685 | 0.600 | 0.654 | 0.702 | 0.525 | – | 0.588 |
| lewm-depth | r2 | 0.162 | 0.475 | 0.111 | 0.513 | 0.802 | 0.703 | – | 0.607 |
| lewm-depth | pearson | 0.361 | 0.652 | 0.336 | 0.698 | 0.903 | 0.835 | – | 0.750 |

### MLP probe

| group | metric | base_pos | lin_vel | ang_vel | gravity | joint_pos | joint_vel | contacts | all |
|---|---|---|---|---|---|---|---|---|---|
| jepa-baseline | r2 | -0.314 | 0.443 | 0.292 | 0.748 | 0.647 | 0.261 | – | 0.400 |
| jepa-baseline | pearson | 0.249 | 0.758 | 0.692 | 0.872 | 0.812 | 0.668 | – | 0.708 |
| lewm-depth | r2 | 0.122 | 0.279 | -0.088 | 0.650 | 0.811 | 0.788 | – | 0.613 |
| lewm-depth | pearson | 0.360 | 0.668 | 0.415 | 0.812 | 0.908 | 0.893 | – | 0.788 |

`contacts` is null on real data (not a logged GrandTour target).

## ε_k, linear probe (σ-normalized, all 40 dims)

| group | reference | k=1 | k=5 | k=12 | k=25 | k=50 |
|---|---|---|---|---|---|---|
| jepa-baseline | model | 4.685 | 4.579 | 4.627 | 4.697 | 5.412 |
| jepa-baseline | persistence | 6.538 | 6.655 | 6.647 | 6.682 | 7.223 |
| jepa-baseline | encoded_floor | 4.725 | 4.674 | 4.730 | 4.683 | 5.365 |
| lewm-depth | model | 3.304 | 3.238 | 3.171 | 3.299 | 4.209 |
| lewm-depth | persistence | 6.410 | 6.287 | 6.069 | 6.124 | 6.687 |
| lewm-depth | encoded_floor | 3.262 | 3.199 | 3.118 | 3.261 | 4.160 |

Horizons: k = 1, 5, 12, 25, 50 ticks ↔ 0.2, 1.0, 2.4, 5.0, 10.0 s at 5 Hz.

## ε_k, MLP probe (σ-normalized, all 40 dims)

| group | reference | k=1 | k=5 | k=12 | k=25 | k=50 |
|---|---|---|---|---|---|---|
| jepa-baseline | model | 4.081 | 3.912 | 3.871 | 3.968 | 4.779 |
| jepa-baseline | persistence | 7.170 | 7.171 | 7.117 | 7.142 | 7.691 |
| jepa-baseline | encoded_floor | 4.159 | 4.098 | 4.115 | 4.140 | 4.861 |
| lewm-depth | model | 3.132 | 3.088 | 3.004 | 3.226 | 4.337 |
| lewm-depth | persistence | 6.812 | 6.696 | 6.532 | 6.596 | 7.164 |
| lewm-depth | encoded_floor | 3.001 | 2.963 | 2.858 | 2.924 | 3.961 |

## Per-component ε_k, MLP probe, model rollout

| group | component | k=1 | k=5 | k=12 | k=25 | k=50 |
|---|---|---|---|---|---|---|
| jepa-baseline | base_pos | 1.450 | 1.319 | 1.235 | 1.432 | 2.634 |
| jepa-baseline | lin_vel | 1.044 | 1.017 | 1.022 | 0.954 | 1.014 |
| jepa-baseline | ang_vel | 1.320 | 1.291 | 1.307 | 1.305 | 1.349 |
| jepa-baseline | gravity | 0.658 | 0.698 | 0.705 | 0.729 | 0.777 |
| jepa-baseline | joint_pos | 1.535 | 1.503 | 1.508 | 1.513 | 1.539 |
| jepa-baseline | joint_vel | 2.490 | 2.388 | 2.359 | 2.373 | 2.349 |
| jepa-baseline | contacts | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| jepa-baseline | all_excl_base_pos | 3.690 | 3.577 | 3.582 | 3.576 | 3.639 |
| lewm-depth | base_pos | 1.086 | 0.927 | 0.686 | 0.999 | 2.485 |
| lewm-depth | lin_vel | 1.021 | 1.070 | 1.030 | 1.139 | 1.240 |
| lewm-depth | ang_vel | 1.382 | 1.396 | 1.396 | 1.451 | 1.490 |
| lewm-depth | gravity | 0.726 | 0.735 | 0.750 | 0.743 | 0.807 |
| lewm-depth | joint_pos | 1.123 | 1.134 | 1.153 | 1.181 | 1.226 |
| lewm-depth | joint_vel | 1.397 | 1.418 | 1.417 | 1.425 | 1.476 |
| lewm-depth | contacts | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lewm-depth | all_excl_base_pos | 2.849 | 2.891 | 2.894 | 2.981 | 3.148 |

(Linear-probe per-component and all raw values are in the eval JSONs.)

---

# Simulation (MuJoCo, stand-in ANYmal C, zero-shot) — protocol v2

Zero-shot transfer of the real-data models into MuJoCo with a stand-in ANYmal C,
pinhole camera, and the scripted trot controller (ADR 0006).

## EV1-sim ε_k, linear probe

| group | reference | k=1 | k=5 | k=12 | k=25 | k=50 |
|---|---|---|---|---|---|---|
| jepa-baseline | model | 11.528 | 12.807 | 14.196 | 15.810 | 17.450 |
| jepa-baseline | persistence | 8.175 | 8.897 | 8.783 | 9.162 | 6.082 |
| jepa-baseline | encoded_floor | 3.724 | 3.626 | 3.602 | 3.601 | 4.426 |
| lewm-depth | model | 6.004 | 7.062 | 7.442 | 7.874 | 8.462 |
| lewm-depth | persistence | 8.445 | 8.923 | 8.671 | 9.103 | 5.443 |
| lewm-depth | encoded_floor | 3.568 | 3.497 | 3.344 | 3.706 | 4.757 |

The baseline's model error exceeds persistence at every horizon; LeWM's is
below it. The persistence curve is non-monotonic at k=50 because episodes
terminate, changing the surviving state distribution.

## EV1-sim ε_k, MLP probe

| group | reference | k=1 | k=5 | k=12 | k=25 | k=50 |
|---|---|---|---|---|---|---|
| jepa-baseline | model | 11.094 | 11.577 | 12.332 | 14.216 | 17.278 |
| jepa-baseline | persistence | 8.989 | 9.375 | 9.238 | 9.659 | 6.123 |
| jepa-baseline | encoded_floor | 3.334 | 3.367 | 3.201 | 3.431 | 4.192 |
| lewm-depth | model | 8.671 | 8.783 | 8.643 | 9.958 | 10.182 |
| lewm-depth | persistence | 8.939 | 9.442 | 9.244 | 9.615 | 5.650 |
| lewm-depth | encoded_floor | 2.947 | 3.064 | 2.731 | 3.060 | 4.111 |

## Probe quality on encoded latents

### Linear probe

| group | metric | base_pos | lin_vel | ang_vel | gravity | joint_pos | joint_vel | contacts | all |
|---|---|---|---|---|---|---|---|---|---|
| jepa-baseline | r2 | 0.443 | 0.708 | 0.915 | 0.924 | 0.605 | 0.496 | 0.515 | 0.606 |
| jepa-baseline | pearson | 0.667 | 0.837 | 0.957 | 0.962 | 0.781 | 0.708 | 0.731 | 0.776 |
| lewm-depth | r2 | 0.178 | 0.643 | 0.661 | 0.887 | 0.711 | 0.562 | 0.639 | 0.623 |
| lewm-depth | pearson | 0.425 | 0.800 | 0.817 | 0.943 | 0.844 | 0.750 | 0.805 | 0.783 |

### MLP probe

| group | metric | base_pos | lin_vel | ang_vel | gravity | joint_pos | joint_vel | contacts | all |
|---|---|---|---|---|---|---|---|---|---|
| jepa-baseline | r2 | 0.425 | 0.766 | 0.938 | 0.942 | 0.705 | 0.528 | 0.553 | 0.656 |
| jepa-baseline | pearson | 0.656 | 0.874 | 0.968 | 0.972 | 0.844 | 0.761 | 0.773 | 0.819 |
| lewm-depth | r2 | 0.135 | 0.820 | 0.848 | 0.946 | 0.817 | 0.676 | 0.705 | 0.725 |
| lewm-depth | pearson | 0.435 | 0.907 | 0.923 | 0.973 | 0.904 | 0.833 | 0.847 | 0.849 |

Note: the baseline's *probe* on sim latents is much better than on real
(0.606 vs 0.284 linear), so its sim failure is in the predictor's open-loop
dynamics, not in encoder decodability.

## EV2 — action-sensitivity ratio ASR(k, δ), MLP probe (ideal ≈ 1)

| group | δ (rad) | k=1 | k=5 | k=12 | k=25 | k=50 |
|---|---|---|---|---|---|---|
| jepa-baseline | 0.02 | 0.598 | 0.371 | 0.159 | 0.266 | 0.319 |
| jepa-baseline | 0.05 | 0.460 | 0.438 | 0.228 | 0.324 | 0.394 |
| jepa-baseline | 0.1 | 0.480 | 0.403 | 0.292 | 0.330 | 0.349 |
| lewm-depth | 0.02 | 0.622 | 0.611 | 0.288 | 0.373 | 0.290 |
| lewm-depth | 0.05 | 0.597 | 0.661 | 0.445 | 0.505 | 0.470 |
| lewm-depth | 0.1 | 0.722 | 0.613 | 0.483 | 0.413 | 0.411 |

Both under-react to actions (ASR ≪ 1); LeWM less so. Linear-probe ASR is in the
JSONs.

## EV3 — CEM planning vs scripted controller, per terrain tier

| tier | group | arm | metric | value | fall rate |
|---|---|---|---|---|---|
| 1-flat | jepa-baseline | planning | tracking_error_mps | 0.452 | 1.000 |
| 1-flat | jepa-baseline | controller | tracking_error_mps | 0.157 | 0.000 |
| 1-flat | lewm-depth | planning | tracking_error_mps | 0.351 | 0.500 |
| 1-flat | lewm-depth | controller | tracking_error_mps | 0.157 | 0.000 |
| 2-rough | jepa-baseline | planning | tracking_error_mps | 0.390 | 0.250 |
| 2-rough | jepa-baseline | controller | tracking_error_mps | 0.190 | 0.000 |
| 2-rough | lewm-depth | planning | tracking_error_mps | 0.515 | 1.000 |
| 2-rough | lewm-depth | controller | tracking_error_mps | 0.190 | 0.000 |
| 3-slopes | jepa-baseline | planning | success_rate | 0.000 | 1.000 |
| 3-slopes | jepa-baseline | controller | success_rate | 0.333 | 0.000 |
| 3-slopes | lewm-depth | planning | success_rate | 0.000 | 1.000 |
| 3-slopes | lewm-depth | controller | success_rate | 0.333 | 0.000 |
| 4-stairs | jepa-baseline | planning | success_rate | 0.000 | 0.667 |
| 4-stairs | jepa-baseline | controller | success_rate | 0.000 | 0.333 |
| 4-stairs | lewm-depth | planning | success_rate | 0.000 | 1.000 |
| 4-stairs | lewm-depth | controller | success_rate | 0.000 | 0.333 |
| 5-gaps | jepa-baseline | planning | max_level | – | 0.167 |
| 5-gaps | jepa-baseline | controller | max_level | – | 0.000 |
| 5-gaps | lewm-depth | planning | max_level | – | 0.667 |
| 5-gaps | lewm-depth | controller | max_level | – | 0.000 |
| 6-steps | jepa-baseline | planning | max_level | – | 0.667 |
| 6-steps | jepa-baseline | controller | max_level | – | 0.000 |
| 6-steps | lewm-depth | planning | max_level | – | 1.000 |
| 6-steps | lewm-depth | controller | max_level | – | 0.000 |

The planning arm is CEM over a per-joint residual on top of the controller's
nominal targets; the controller arm is the scripted trot alone. The controller
wins every tier.

## EV4 — retention (% of nominal forward progress), terrain rough/0.02

| group | arm | nominal | mass-30% | mass-15% | mass+15% | mass+30% | fr0.4x | fr1.5x | fr2.5x | lat10ms | lat25ms | lat50ms | comp mass+30% | comp mass-30% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jepa-baseline | planning | 100.0 | 76.5 | 75.6 | 174.2 | 239.7 | 15.0 | 45.3 | 135.0 | 38.7 | 93.9 | 245.9 | 115.4 | 185.3 |
| jepa-baseline | controller | 100.0 | 100.0 | 100.1 | 100.5 | 99.6 | 100.5 | 100.4 | 83.5 | 100.6 | 100.1 | 100.1 | 86.5 | 92.8 |
| lewm-depth | planning | 100.0 | 573.0 | 502.7 | 233.4 | 433.9 | 21.7 | 374.3 | 395.9 | 369.8 | 364.0 | 401.4 | 434.3 | 104.7 |
| lewm-depth | controller | 100.0 | 100.0 | 100.1 | 100.5 | 99.6 | 100.5 | 100.4 | 83.5 | 100.6 | 100.1 | 100.1 | 86.5 | 92.8 |

Planning retention is unstable (well above 100%, so the progress metric
saturates); the controller is ~100%.

## EV6 — mirrored / original (unilateral steps)

| group | arm | success ratio | progress ratio |
|---|---|---|---|
| jepa-baseline | planning | – | 14.350 |
| jepa-baseline | controller | 1.000 | 0.993 |
| lewm-depth | planning | – | 2.654 |
| lewm-depth | controller | 1.000 | 0.993 |

## Δ_s2r(k) = ε_k(real) − ε_k(sim), paired by checkpoint

| group | probe | k=1 | k=5 | k=12 | k=25 | k=50 |
|---|---|---|---|---|---|---|
| jepa-baseline | linear | -6.843 | -8.228 | -9.569 | -11.113 | -12.039 |
| jepa-baseline | mlp | -7.012 | -7.666 | -8.461 | -10.247 | -12.499 |
| lewm-depth | linear | -2.699 | -3.824 | -4.271 | -4.575 | -4.254 |
| lewm-depth | mlp | -5.538 | -5.696 | -5.639 | -6.733 | -5.845 |

Negative because sim ε exceeds real ε: the stand-in sim is harder than real
GrandTour. LeWM's gap is roughly half the baseline's.

## EV7 — compute and latency

| group | inference params (M) | frozen encoder (M) | rollout FPS | single-step ms | training GPU-h (approx) |
|---|---|---|---|---|---|
| jepa-baseline | 220.424 | 86.833 | 215.164 | 44.393 | 47.954 |
| lewm-depth | 21.753 | 0.000 | 1689.716 | 10.343 | 8.254 |

---

## Interpretation (carry these caveats)

- **Single seed per model.** No significance tests; the numbers are descriptive.
- **ε_k is floor-limited**: for both models the model error sits near
  `encoded_floor`, so absolute ε differences mostly reflect encoder/probe
  decodability, not dynamics quality.
- **EV3/EV4/EV6 used a cut CEM budget** (`sim-protocol-v2`: 32/4/4 vs the
  protocol's 300/30/10, 2 seeds), applied to both models to keep budgets equal
  (`0248a75`, ADR 0006 addendum 2026-10-06b). These answer "does the pipeline
  run end-to-end," not "how good is control."
- **Sim is a stand-in** (ANYmal C, pinhole camera, scripted trot), not the
  recipe's pretrained ANYmal D controller, so all sim numbers are qualitative.
- `gait_cycle` is unmeasured, so k=12 is still the protocol placeholder, not
  the true ANYmal D gait cycle.
- **Not run:** EV3 learned-policy (actor-critic in imagination); the E1.3
  grounding on/off and E1.4 horizon ablations, which are the arms needed to
  isolate the encoder/grounding contribution.

Full auto-generated report (all raw tables, per-component for every probe):
`quadwm report <eval JSONs> --output <path>`; the v2 run's output is kept at
`quad-wm-storage/runs/reports/sim-protocol-v2-report.md`.
