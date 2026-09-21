# quad-wm

Collaborative infrastructure for quadruped world-model experiments, starting with Track 1: a depth-and-proprioception JEPA for locomotion.

## Current status

This repository is at the infrastructure-and-data-contract stage. It contains a CPU-only smoke runner, a Slurm entry point, W&B integration hooks, collaboration rules, the research recipes, and a deterministic GrandTour Track 1 consumer. Simulator and ANYmal-D assets are intentionally not installed yet; MuJoCo is the first simulator candidate and Isaac Lab is optional.

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

The smoke test has no PyTorch or simulator dependency. It validates run directories, resolved metadata, deterministic synthetic loss generation, and optional W&B logging.

```bash
python -m quadwm smoke --steps 20 --output-dir runs/local-smoke
python -m pytest -q
```

For Slurm, edit the project and virtual-environment paths in `scripts/slurm/track1_smoke.sbatch` or export them before submission:

```bash
export PROJECT_DIR="$HOME/quad-wm"
export VENV_DIR="$PROJECT_DIR/.venv"
export RUN_ROOT="$HOME/quad-wm-runs"
sbatch scripts/slurm/track1_smoke.sbatch
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
uv sync --extra grandtour
sbatch scripts/slurm/track1_baseline.sbatch
```

Override the allocation or paths without editing the script:

```bash
sbatch --gres=gpu:3 \
  --export=ALL,PROJECT_DIR="$PWD",CONFIG=configs/jepa-wm/baseline.yaml \
  scripts/slurm/track1_baseline.sbatch
```

The encoder checkpoint and feature cache live under the configured external
roots. `quadwm prepare --config configs/jepa-wm/baseline.yaml` can be run
interactively first; it is idempotent.

## GrandTour Track 1 data path

Install the optional reader dependencies on the cluster environment:

```bash
uv sync --extra grandtour
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
