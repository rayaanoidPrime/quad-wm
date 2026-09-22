#!/bin/bash
#SBATCH --job-name=quadwm-jepa
#SBATCH --output=logs/quadwm-jepa-%j.out
#SBATCH --error=logs/quadwm-jepa-%j.err
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/quad-wm}"
PROJECT_PARENT="$(dirname "$PROJECT_DIR")"
STORAGE_ROOT="${QUADWM_STORAGE_ROOT:-$PROJECT_PARENT/quad-wm-storage}"
RUN_ROOT="${RUN_ROOT:-${QUADWM_RUN_ROOT:-$STORAGE_ROOT/runs/$SLURM_JOB_ID}}"
CONFIG="${CONFIG:-configs/jepa-wm/baseline.yaml}"

module purge
module load python/3.11
module load rocm/6.2.4

cd "$PROJECT_DIR"
export GRANDTOUR_ROOT="${GRANDTOUR_ROOT:-$STORAGE_ROOT/data/grandtour}"
export QUADWM_CHECKPOINT_ROOT="${QUADWM_CHECKPOINT_ROOT:-$STORAGE_ROOT/checkpoints}"
export QUADWM_CACHE_ROOT="${QUADWM_CACHE_ROOT:-$STORAGE_ROOT/cache/vjepa21}"
export QUADWM_RUN_ROOT="$RUN_ROOT"
mkdir -p logs "$GRANDTOUR_ROOT" "$QUADWM_CHECKPOINT_ROOT" "$QUADWM_CACHE_ROOT" "$RUN_ROOT"
export QUADWM_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "job=$SLURM_JOB_ID host=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-1} commit=$QUADWM_GIT_COMMIT"
echo "config=$CONFIG storage_root=$STORAGE_ROOT run_root=$RUN_ROOT"

GPUS="${SLURM_GPUS_ON_NODE:-1}"
uv run quadwm prepare --config "$CONFIG"
torchrun --standalone --nproc_per_node="$GPUS" -m quadwm train --config "$CONFIG"
