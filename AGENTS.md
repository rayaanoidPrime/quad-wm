# Agent and collaborator contract

## Scope

This repository is research infrastructure for Track 1: a depth + proprioception JEPA world model for quadruped locomotion.

## Before changing code

1. Read `MISSION.md`, the relevant recipe in `recipes/`, and the shared evaluation protocol.
2. Check whether the change affects reproducibility, data splits, metrics, or checkpoint compatibility.
3. Keep raw data, secrets, checkpoints, and generated logs outside Git.

## Change rules

- Prefer small, reviewable pull requests.
- Put experiment behavior in versioned config, not hidden command-line defaults.
- Keep simulator, dataset, model, training, and evaluation boundaries explicit.
- Add or update a focused test for deterministic data/config/metric behavior.
- Never silently change a mission split, normalization, horizon set, or metric definition.
- Record non-obvious research decisions in `docs/adr/`.

## Verification

At minimum, run the CPU-only unit tests (`uv run pytest -q`) and, on a GPU node with materialized GrandTour data, the end-to-end smoke (`uv run quadwm smoke`). The smoke is the same pipeline as `train` on the tiny config, so it surfaces download/data/cache/encoder/eval failures before a full run. GPU or simulator tests must be marked separately because CI does not have MI300X/Isaac Lab access.

## Outputs

Every training run should record its Git commit, resolved configuration, seed, Slurm job ID, host, visible GPU information, and W&B run URL when online logging is enabled.
