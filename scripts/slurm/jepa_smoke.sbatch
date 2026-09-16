#!/bin/bash
#SBATCH --job-name=quadwm-jepa-smoke
#SBATCH --output=jepa-smoke-%j.out
#SBATCH --error=jepa-smoke-%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail

# Override these at submission time when the cluster checkout differs.
PROJECT_DIR="${PROJECT_DIR:-$HOME/quad-wm}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
RUN_ROOT="${RUN_ROOT:-$HOME/quad-wm-runs}"

module purge
module load python/3.11
module load rocm/6.2.4

cd "$PROJECT_DIR"

export QUADWM_RUN_ROOT="$RUN_ROOT/slurm/$SLURM_JOB_ID"
export QUADWM_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "project=$PROJECT_DIR"
echo "run_root=$QUADWM_RUN_ROOT"
echo "git_commit=$QUADWM_GIT_COMMIT"

uv run quadwm smoke --config configs/jepa-wm/smoke.yaml
