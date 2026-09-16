# quad-wm

Collaborative infrastructure for quadruped world-model experiments, starting with Track 1: a depth-and-proprioception JEPA for locomotion.

## Current status

This repository is at the infrastructure-scaffold stage. It contains a CPU-only smoke runner, a Slurm entry point, W&B integration hooks, collaboration rules, and the research recipes. Simulator and ANYmal-D assets are intentionally not installed yet; MuJoCo is the first simulator candidate and Isaac Lab is optional.

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

To publish metrics, authenticate once on the cluster and submit with `WANDB_MODE=online` and `WANDB_PROJECT=<project-name>`. Never put the API key in Git or a batch script.

## Research storage boundary

Keep the Git checkout small. Put GrandTour, Isaac assets, caches, checkpoints, and run artifacts under explicit external roots such as:

```text
$SCRATCH/quad-wm/data/grandtour
$SCRATCH/quad-wm/runs
$SCRATCH/quad-wm/checkpoints
```

The exact `$SCRATCH` path is cluster-specific and must be confirmed on the machine.
