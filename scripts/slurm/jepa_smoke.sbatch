#!/bin/bash
#SBATCH --job-name=quadwm-jepa-smoke
#SBATCH --output=logs/jepa-smoke-%j.out
#SBATCH --error=logs/jepa-smoke-%j.err
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

# The cluster's gres.conf maps the 8 XCD partitions of physical GPU 0 as
# "GPUs", so SLURM_JOB_GPUS is not a usable physical GPU index; Slurm's device
# isolation also hides physical GPU 0 from what the job's runtime can open.
# Pick the freest GPU(s) the job can actually see, instead of trusting the
# allocation. Clear the visibility vars first so the runtime enumerates them all.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES="$(
  GPUS="${SLURM_GPUS_ON_NODE:-1}" "$PROJECT_DIR/.venv/bin/python3" - <<'PY'
import os, torch
want = int(os.environ.get("GPUS", "1"))
ranked = sorted(range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.mem_get_info(i)[0], reverse=True)
print(",".join(map(str, sorted(ranked[:want]))))
PY
)"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
echo "[$(date -Is)] gpus: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
rocm-smi --showmemuse --showpids 2>/dev/null || true

export GRANDTOUR_ROOT="${GRANDTOUR_ROOT:-$STORAGE_ROOT/data/grandtour}"
export QUADWM_RUN_ROOT="$RUN_ROOT"
export QUADWM_CHECKPOINT_ROOT="${QUADWM_CHECKPOINT_ROOT:-$STORAGE_ROOT/checkpoints}"
export QUADWM_CACHE_ROOT="${QUADWM_CACHE_ROOT:-$STORAGE_ROOT/cache/vjepa21-smoke}"
mkdir -p logs "$GRANDTOUR_ROOT" "$QUADWM_CHECKPOINT_ROOT" "$QUADWM_CACHE_ROOT" "$RUN_ROOT"
export QUADWM_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

log() { echo "[$(date -Is)] $*"; }

log "stage=job status=started id=$SLURM_JOB_ID host=$(hostname)"
log "project=$PROJECT_DIR config=$CONFIG"
log "grandtour_root=$GRANDTOUR_ROOT run_root=$QUADWM_RUN_ROOT"
log "git_commit=$QUADWM_GIT_COMMIT"

log "stage=prepare status=starting"
uv run quadwm prepare --config "$CONFIG"
log "stage=prepare status=complete"
log "stage=train status=starting"
uv run torchrun --standalone --nproc_per_node=1 -m quadwm train --config "$CONFIG"
log "stage=train status=complete"
log "stage=job status=complete"
