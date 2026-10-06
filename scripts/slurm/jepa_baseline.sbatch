#!/bin/bash
#SBATCH --job-name=quadwm-jepa
#SBATCH --output=logs/jepa-%j.out
#SBATCH --error=logs/jepa-%j.err
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/quad-wm}"
PROJECT_PARENT="$(dirname "$PROJECT_DIR")"
STORAGE_ROOT="${QUADWM_STORAGE_ROOT:-$PROJECT_PARENT/quad-wm-storage}"
# Stable run dir (not per-job-id) so checkpoints survive the walltime and the
# next submission resumes from last.pt. Per-job stdout/stderr still use %j.
RUN_ROOT="${RUN_ROOT:-${QUADWM_RUN_ROOT:-$STORAGE_ROOT/runs/jepa-baseline}}"
CONFIG="${CONFIG:-configs/jepa-wm/baseline.yaml}"

module purge
module load python/3.11
module load rocm/6.2.4

cd "$PROJECT_DIR"

# GPU_SELECT=freest (default, docs/adr/0001): the cluster's gres.conf once
# mapped the 8 XCD partitions of physical GPU 0 as "GPUs", so SLURM_JOB_GPUS
# was not a usable physical index; pick the freest GPU(s) the job can see.
# GPU_SELECT=slurm: trust SLURM_JOB_GPUS, as the sim eval does since 8efbd77.
# Use it when several jobs start together (scripts/slurm/lewm_pipeline.sh):
# "freest" races and lands simultaneous jobs on the same card.
GPU_SELECT="${GPU_SELECT:-freest}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
if [[ "$GPU_SELECT" == slurm && -n "${SLURM_JOB_GPUS:-}" ]]; then
  export ROCR_VISIBLE_DEVICES="${SLURM_JOB_GPUS// /,}"
else
  export ROCR_VISIBLE_DEVICES="$(
    GPUS="${SLURM_GPUS_ON_NODE:-1}" "$PROJECT_DIR/.venv/bin/python3" - <<'PY'
import os, torch
want = int(os.environ.get("GPUS", "1"))
ranked = sorted(range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.mem_get_info(i)[0], reverse=True)
print(",".join(map(str, sorted(ranked[:want]))))
PY
  )"
fi
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
echo "[$(date -Is)] gpus: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
rocm-smi --showmemuse --showpids 2>/dev/null || true
export GRANDTOUR_ROOT="${GRANDTOUR_ROOT:-$STORAGE_ROOT/data/grandtour}"
export QUADWM_CHECKPOINT_ROOT="${QUADWM_CHECKPOINT_ROOT:-$STORAGE_ROOT/checkpoints}"
export QUADWM_CACHE_ROOT="${QUADWM_CACHE_ROOT:-$STORAGE_ROOT/cache/vjepa21}"
export QUADWM_RUN_ROOT="$RUN_ROOT"
# Sim fine-tuning configs read episodes rendered by jepa_sim_collect.sbatch here.
export QUADWM_SIM_CACHE_ROOT="${QUADWM_SIM_CACHE_ROOT:-$STORAGE_ROOT/cache/sim-episodes}"
mkdir -p logs "$GRANDTOUR_ROOT" "$QUADWM_CHECKPOINT_ROOT" "$QUADWM_CACHE_ROOT" "$RUN_ROOT"
export QUADWM_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

log() { echo "[$(date -Is)] $*"; }

log "stage=job status=started id=$SLURM_JOB_ID host=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-1}"
log "config=$CONFIG grandtour_root=$GRANDTOUR_ROOT storage_root=$STORAGE_ROOT run_root=$RUN_ROOT"
log "git_commit=$QUADWM_GIT_COMMIT"
log "missions=${GRANDTOUR_MISSIONS:-all materialized missions}"

GPUS="${SLURM_GPUS_ON_NODE:-1}"
log "stage=prepare status=starting"
uv run quadwm prepare --config "$CONFIG"
log "stage=prepare status=complete"
log "stage=train status=starting gpus=$GPUS"
uv run torchrun --standalone --nproc_per_node="$GPUS" -m quadwm train --config "$CONFIG"
log "stage=train status=complete"
log "stage=job status=complete"
