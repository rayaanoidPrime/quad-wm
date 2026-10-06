#!/bin/bash
#SBATCH --job-name=quadwm-sim-collect
#SBATCH --output=logs/sim-collect-%j.out
#SBATCH --error=logs/sim-collect-%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

# Render and cache simulated episodes once, before any job that reads them:
# the sim fine-tuning set (training never renders) and the EV1-sim episodes
# (so parallel sim-eval shards do not all render the same missing files).
# Idempotent: cached sets are skipped. The GPU is only for EGL rendering;
# without one, Slurm's device cgroup hides /dev/dri. Example:
#   SIM_COLLECT_CONFIGS="configs/sim/finetune_episodes.yaml configs/eval/sim.yaml" \
#     sbatch scripts/slurm/jepa_sim_collect.sbatch
PROJECT_DIR="${PROJECT_DIR:-$HOME/quad-wm}"
PROJECT_PARENT="$(dirname "$PROJECT_DIR")"
STORAGE_ROOT="${QUADWM_STORAGE_ROOT:-$PROJECT_PARENT/quad-wm-storage}"
SIM_COLLECT_CONFIGS="${SIM_COLLECT_CONFIGS:-configs/sim/finetune_episodes.yaml configs/eval/sim.yaml}"

module purge
module load python/3.11
module load rocm/6.2.4

cd "$PROJECT_DIR"

# Same GPU handling as the sim eval (8efbd77): trust Slurm's assignment.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  export ROCR_VISIBLE_DEVICES="${SLURM_JOB_GPUS// /,}"
fi
export QUADWM_SIM_ASSETS="${QUADWM_SIM_ASSETS:-$STORAGE_ROOT/assets}"
export QUADWM_SIM_CACHE_ROOT="${QUADWM_SIM_CACHE_ROOT:-$STORAGE_ROOT/cache/sim-episodes}"
# Headless rendering. If EGL is unavailable on the node, try MUJOCO_GL=osmesa.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p logs "$QUADWM_SIM_ASSETS" "$QUADWM_SIM_CACHE_ROOT"
export QUADWM_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

log() { echo "[$(date -Is)] $*"; }

log "stage=job status=started id=$SLURM_JOB_ID host=$(hostname)"
log "configs=$SIM_COLLECT_CONFIGS cache_root=$QUADWM_SIM_CACHE_ROOT mujoco_gl=$MUJOCO_GL git_commit=$QUADWM_GIT_COMMIT"
for config in $SIM_COLLECT_CONFIGS; do
  log "stage=sim_collect status=starting config=$config"
  uv run --extra sim quadwm sim-collect --sim-eval-config "$config"
  log "stage=sim_collect status=complete config=$config"
done
log "stage=job status=complete"
