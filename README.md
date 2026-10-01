# quad-wm

Collaborative infrastructure for quadruped world-model experiments, starting with Track 1: a depth-and-proprioception JEPA for locomotion.

## Current status

This repository is at the infrastructure-and-data-contract stage. It contains an end-to-end smoke runner, a Slurm entry point, W&B integration hooks, collaboration rules, the research recipes, and a deterministic GrandTour Track 1 consumer. Simulator and ANYmal-D assets are intentionally not installed yet; MuJoCo is the first simulator candidate and Isaac Lab is optional.

The first simulator gate is a headless MuJoCo physics/rendering smoke test on the target node. The code keeps simulation behind an adapter so the same training/evaluation infrastructure can later consume MuJoCo, Isaac Lab, or another simulator without rewriting the JEPA stack.

## Layout

```text
.
├── configs/jepa-wm/      Versioned Track 1 experiment configuration
├── docs/adr/             Non-obvious infrastructure/research decisions
├── lessons/              Short hands-on teaching units
├── recipes/              Research recipes and shared evaluation protocol
├── reference/            Printable quick references
├── scripts/slurm/        Cluster entry points
├── src/quadwm/           Importable package used by jobs and notebooks
└── tests/                CPU-safe tests
```

## First smoke test

`quadwm smoke` runs the *entire* pipeline on the tiny smoke config -- download, data sync, token cache, V-JEPA encoder, one training step, checkpoint save, a held-out eval pass, and the protocol eval (`configs/eval/smoke.yaml`) on the checkpoint it just wrote. It is the same `train` code path, so configuration, data, and eval errors surface before a full training run. It needs an allocated GPU and materialized GrandTour data; the CPU-only check is the unit-test suite.

```bash
uv run quadwm smoke            # == quadwm train --config configs/jepa-wm/baseline_smoke.yaml
uv run pytest -q               # CPU-only unit tests
```

For Slurm, edit the project and virtual-environment paths in `scripts/slurm/jepa_smoke.sbatch` or export them before submission:

```bash
export PROJECT_DIR="$HOME/quad-wm"
export VENV_DIR="$PROJECT_DIR/.venv"
export RUN_ROOT="$HOME/quad-wm-runs"
sbatch scripts/slurm/jepa_smoke.sbatch
squeue --me
tail -f "$RUN_ROOT/slurm/<job-id>.out"
```

To publish metrics, authenticate once on the cluster, then submit with `QUADWM_WANDB_ENABLED=true` and `WANDB_MODE=online`. The config supplies the project/name; never put the API key in Git or a batch script.

## Research storage boundary

Keep the Git checkout small. Put GrandTour, Isaac assets, caches, checkpoints, and run artifacts under explicit external roots such as:

```text
$SCRATCH/quad-wm/data/grandtour
$SCRATCH/quad-wm/runs
$SCRATCH/quad-wm/checkpoints
```

The exact `$SCRATCH` path is cluster-specific and must be confirmed on the machine.

## Train the baseline JEPA world model

The baseline uses RGB, a 5 Hz world-model tick, 10 stacked 50 Hz joint-position
commands, a 7-tick context, and a 6-step rollout. The training command caches
frozen V-JEPA 2.1 tokens locally when the disk estimate fits; otherwise it
streams encoder features without writing a cache.

```bash
uv sync
sbatch scripts/slurm/jepa_baseline.sbatch
```

Override the allocation or paths without editing the script:

```bash
sbatch --gres=gpu:3 \
  --export=ALL,PROJECT_DIR="$PWD",CONFIG=configs/jepa-wm/baseline.yaml \
  scripts/slurm/jepa_baseline.sbatch
```

The encoder checkpoint and feature cache live under the configured external
roots. `uv run quadwm prepare --config configs/jepa-wm/baseline.yaml` can be run
interactively first; it is idempotent.

## Evaluate trained runs (shared protocol)

`quadwm eval` scores one checkpoint under `recipes/shared_evaluation_protocol.md`
on the run's own held-out missions (docs/adr/0005): linear and MLP state
probes fit on the probe-fit missions, probe R² / Pearson r, EV5 ε_k at
k ∈ {1, 5, 12, 25, 50} with per-component breakdown, persistence and
probe-floor references, the measured gait cycle, and EV7 compute/latency.
It writes `<run_root>/<name>/eval/<ckpt>-<eval name>.json`. The simulated
evals run separately (below).

Every model compared must use the same `configs/eval/protocol.yaml`, and
both camera topics must be materialized under `GRANDTOUR_ROOT`: windows are
anchored on moments valid for the RGB *and* depth camera so both recipes are
scored on the same data.

```bash
# Use the CONFIG and RUN_ROOT the run trained with.
CONFIG=configs/jepa-wm/baseline.yaml RUN_ROOT=$STORAGE_ROOT/runs/jepa-baseline   sbatch scripts/slurm/jepa_eval.sbatch
CONFIG=configs/jepa-wm/lewm_depth.yaml RUN_ROOT=$STORAGE_ROOT/runs/lewm-depth   sbatch scripts/slurm/jepa_eval.sbatch

# Aggregate across seeds and models (mean ± std, Mann-Whitney U + Holm).
uv run quadwm report $STORAGE_ROOT/runs/*/*/eval/last-*.json --output report.md
```

### Simulated evals (EV1-sim, Δ_s2r, EV2, EV3, EV4, EV6)

`quadwm sim-eval` runs the rest of the protocol in MuJoCo with a scripted trot
controller (docs/adr/0006): EV1-sim with the same probe procedure, EV2
action sensitivity from simulator replays, EV3 CEM planning over the terrain
suite next to the controller alone, EV4 dynamics-shift retention, and EV6
mirrored terrain. Models are trained on GrandTour only, so this is zero-shot
transfer to a stand-in ANYmal C; read the limits in the ADR before quoting it.

```bash
# Once: MuJoCo + robot_descriptions. The ANYmal C MJCF (MuJoCo Menagerie, pinned
# commit, ~2 GB) is fetched into $QUADWM_SIM_ASSETS on first use.
uv sync --extra sim
# Optional, CPU node: render the shared EV1-sim episodes ahead of time.
uv run --extra sim quadwm sim-collect
# Per model; ONLY=ev3 etc. splits the work.
CONFIG=configs/jepa-wm/lewm_depth.yaml RUN_ROOT=$STORAGE_ROOT/runs/lewm-depth   sbatch scripts/slurm/jepa_sim_eval.sbatch
```

Pass real and sim JSONs together to `quadwm report` to get Δ_s2r(k).

`quadwm report` flags runs whose eval config, horizons, windows, or mission
splits differ; protocol §11 needs at least 3 seeds per model
before any comparison is called a result.

## GrandTour Track 1 data path

Install the project dependencies on the cluster environment:

```bash
uv sync
```

GrandTour is downloaded as gated topic archives and materialized outside the
repository. Authenticate with the Hugging Face CLI first, then download one
mission:

```bash
export GRANDTOUR_ROOT="$SCRATCH/quad-wm/data/grandtour"
bash scripts/hf/download_grandtour_mission.sh 2024-11-02-17-18-32 "$GRANDTOUR_ROOT"
uv run quadwm grandtour inspect --root "$GRANDTOUR_ROOT" \
  --mission 2024-11-02-17-18-32
uv run quadwm grandtour consume --root "$GRANDTOUR_ROOT" \
  --mission 2024-11-02-17-18-32
```

The consumer anchors on depth timestamps and produces lazy samples with
33-dimensional proprioception, the shared 40-dimensional physical state, and
12 commanded joint-position actions. The versioned config requests a 10-frame
proprioceptive history for the Track 1 encoder. It drops samples outside the
configured 50 ms synchronization window. The adapter is NumPy-only so it can
be used by future PyTorch dataloaders, Slurm jobs, and cloud notebooks without
changing the data contract.
