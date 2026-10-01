#!/bin/bash
#SBATCH --job-name=quadwm-eval
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/quad-wm}"
PROJECT_PARENT="$(dirname "$PROJECT_DIR")"
STORAGE_ROOT="${QUADWM_STORAGE_ROOT:-$PROJECT_PARENT/quad-wm-storage}"
# Same RUN_ROOT and CONFIG the run trained with, so the eval finds its
# last.pt, splits.json and normalization. Example:
#   CONFIG=configs/jepa-wm/lewm_depth.yaml RUN_ROOT=$STORAGE_ROOT/runs/lewm-depth sbatch scripts/slurm/jepa_eval.sbatch
RUN_ROOT="${RUN_ROOT:-${QUADWM_RUN_ROOT:-$STORAGE_ROOT/runs/jepa-baseline}}"
CONFIG="${CONFIG:-configs/jepa-wm/baseline.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval/protocol.yaml}"

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
export QUADWM_CHECKPOINT_ROOT="${QUADWM_CHECKPOINT_ROOT:-$STORAGE_ROOT/checkpoints}"
export QUADWM_CACHE_ROOT="${QUADWM_CACHE_ROOT:-$STORAGE_ROOT/cache/vjepa21}"
export QUADWM_RUN_ROOT="$RUN_ROOT"
mkdir -p logs "$GRANDTOUR_ROOT" "$QUADWM_CHECKPOINT_ROOT" "$QUADWM_CACHE_ROOT" "$RUN_ROOT"
export QUADWM_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

log() { echo "[$(date -Is)] $*"; }

log "stage=job status=started id=$SLURM_JOB_ID host=$(hostname)"
log "config=$CONFIG eval_config=$EVAL_CONFIG checkpoint=${CHECKPOINT:-<run>/last.pt}"
log "grandtour_root=$GRANDTOUR_ROOT run_root=$RUN_ROOT git_commit=$QUADWM_GIT_COMMIT"

log "stage=eval status=starting"
uv run quadwm eval --config "$CONFIG" --eval-config "$EVAL_CONFIG" ${CHECKPOINT:+--checkpoint "$CHECKPOINT"}
log "stage=eval status=complete"
log "stage=job status=complete"
