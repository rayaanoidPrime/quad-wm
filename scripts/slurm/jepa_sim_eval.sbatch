#!/bin/bash
#SBATCH --job-name=quadwm-sim-eval
#SBATCH --output=logs/sim-eval-%j.out
#SBATCH --error=logs/sim-eval-%j.err
#SBATCH --time=23:00:00
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
#   CONFIG=configs/jepa-wm/lewm_depth.yaml RUN_ROOT=$STORAGE_ROOT/runs/lewm-depth sbatch scripts/slurm/jepa_sim_eval.sbatch
# Optional: ONLY=ev1,ev2 to run a subset (probes are always fit).
#
# Sharded across GPUs as a job array: one GPU per shard, each writing its own
# <ckpt>-<sim name>-<evals>.json, which `quadwm report` merges. EV3 and EV4 are
# the planning-heavy evals, so the default split gives each its own GPU:
#   CONFIG=... RUN_ROOT=... sbatch --array=0-2 scripts/slurm/jepa_sim_eval.sbatch
# Override the split with space-separated ONLY_SETS (one entry per array index).
# Optional, once per sim config: render the shared EV1 episodes first so the
# shards do not all render them: uv run --extra sim quadwm sim-collect
RUN_ROOT="${RUN_ROOT:-${QUADWM_RUN_ROOT:-$STORAGE_ROOT/runs/jepa-baseline}}"
CONFIG="${CONFIG:-configs/jepa-wm/baseline.yaml}"
SIM_EVAL_CONFIG="${SIM_EVAL_CONFIG:-configs/eval/sim.yaml}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${ONLY:-}" ]]; then
  read -r -a SHARDS <<< "${ONLY_SETS:-ev1,ev2,ev6 ev3 ev4}"
  if (( SLURM_ARRAY_TASK_ID >= ${#SHARDS[@]} )); then
    echo "array index $SLURM_ARRAY_TASK_ID has no shard in ONLY_SETS=(${SHARDS[*]})" >&2
    exit 1
  fi
  ONLY="${SHARDS[$SLURM_ARRAY_TASK_ID]}"
fi

module purge
module load python/3.11
module load rocm/6.2.4

cd "$PROJECT_DIR"

# 8 real MI300X cards here, with distinct serials; Slurm hands each array task
# a distinct GPU. Trust that assignment. Re-picking the "freest" GPU races when
# array tasks start together and lands them all on one card (seen: 5 shards on
# GPU 6). Fall back to the freest GPU only if Slurm gave us no index.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
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
mkdir -p logs "$GRANDTOUR_ROOT" "$QUADWM_CHECKPOINT_ROOT" "$QUADWM_CACHE_ROOT" "$RUN_ROOT"
export QUADWM_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
# MuJoCo assets live outside Git (configs/sim/mujoco_anymal.yaml); episodes
# are rendered once and shared by every model evaluated with the same config.
export QUADWM_SIM_ASSETS="${QUADWM_SIM_ASSETS:-$STORAGE_ROOT/assets}"
export QUADWM_SIM_CACHE_ROOT="${QUADWM_SIM_CACHE_ROOT:-$STORAGE_ROOT/cache/sim-episodes}"
# Headless rendering. If EGL is unavailable on the node, try MUJOCO_GL=osmesa.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# First use fetches the ANYmal C MJCF into $QUADWM_SIM_ASSETS via robot_descriptions
# (needs network once); later jobs reuse it. Prefetch: uv run --extra sim quadwm sim-smoke

log() { echo "[$(date -Is)] $*"; }

log "stage=job status=started id=$SLURM_JOB_ID host=$(hostname)"
log "config=$CONFIG sim_eval_config=$SIM_EVAL_CONFIG checkpoint=${CHECKPOINT:-<run>/last.pt} only=${ONLY:-all}"
log "run_root=$RUN_ROOT sim_assets=$QUADWM_SIM_ASSETS mujoco_gl=$MUJOCO_GL git_commit=$QUADWM_GIT_COMMIT"

log "stage=sim_eval status=starting"
uv run --extra sim quadwm sim-eval --config "$CONFIG" --sim-eval-config "$SIM_EVAL_CONFIG"   ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} ${ONLY:+--only "$ONLY"}
log "stage=sim_eval status=complete"
log "stage=job status=complete"
