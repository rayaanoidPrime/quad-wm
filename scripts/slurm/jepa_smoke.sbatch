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
PROJECT_PARENT="$(dirname "$PROJECT_DIR")"
STORAGE_ROOT="${QUADWM_STORAGE_ROOT:-$PROJECT_PARENT/quad-wm-storage}"
RUN_ROOT="${RUN_ROOT:-${QUADWM_RUN_ROOT:-$STORAGE_ROOT/runs/smoke-$SLURM_JOB_ID}}"
CONFIG="${CONFIG:-configs/jepa-wm/baseline_smoke.yaml}"

module purge
module load python/3.11
module load rocm/6.2.4

cd "$PROJECT_DIR"

export GRANDTOUR_ROOT="${GRANDTOUR_ROOT:-$STORAGE_ROOT/data/grandtour}"
export QUADWM_RUN_ROOT="$RUN_ROOT"
export QUADWM_CHECKPOINT_ROOT="${QUADWM_CHECKPOINT_ROOT:-$STORAGE_ROOT/checkpoints}"
export QUADWM_CACHE_ROOT="${QUADWM_CACHE_ROOT:-$STORAGE_ROOT/cache/vjepa21-smoke}"
mkdir -p logs "$GRANDTOUR_ROOT" "$QUADWM_CHECKPOINT_ROOT" "$QUADWM_CACHE_ROOT" "$RUN_ROOT"
export QUADWM_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "project=$PROJECT_DIR"
echo "config=$CONFIG"
echo "grandtour_root=$GRANDTOUR_ROOT"
echo "run_root=$QUADWM_RUN_ROOT"
echo "git_commit=$QUADWM_GIT_COMMIT"

uv run quadwm prepare --config "$CONFIG"
uv run torchrun --standalone --nproc_per_node=1 -m quadwm train --config "$CONFIG"
