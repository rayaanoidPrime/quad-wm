# ADR 0001 — Three-way mission split (train / probe-fit / eval)

- Status: accepted
- Date: 2026-09-27

## Context

The shared evaluation protocol (§1.3, rule 4) requires probes to be fit on
missions the world model never trained on, and scored on a third,
disjoint set. The baseline recipe (E1.2) instead says to fit probes on
train-split missions. The two conflict; the shared protocol wins because
it governs cross-track comparison.

## Decision

`split_mission_names(names, seed, eval_fraction, probe_fraction)` returns
`(train, eval, probe)`. The seeded shuffle is unchanged and eval missions
are still its first `eval_fraction` slice, so **the held-out eval set is
identical to the previous two-way split**. Probe-fit missions are the next
`probe_fraction` slice, taken out of the former training pool.

`configs/data/baseline.yaml` sets `eval_fraction: 0.2`, `probe_fraction: 0.2`
(about 29 / 10 / 10 of the 49 GrandTour missions). Every run writes
`splits.json` to its run directory.

## Consequences

- Runs trained before this change used the probe-fit missions for training.
  Their checkpoints must not be used for protocol probe results; retrain.
- Baseline recipe E1.2's "probes fit on train-split missions" is superseded
  for protocol-facing numbers.
