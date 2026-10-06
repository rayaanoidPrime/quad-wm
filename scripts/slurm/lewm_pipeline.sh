#!/bin/bash
# Submit the docs/adr/0007 LeWM arms, train -> eval, as one Slurm dependency
# graph. Run it on the login node from the checkout (it only calls sbatch):
#
#   bash scripts/slurm/lewm_pipeline.sh
#
#   sim-collect ─────────────────────────────────────────────┐
#   per seed:  grounded train ──> eval + sim-eval[0-2]       │
#              residual train ──> eval + sim-eval[0-2]       │
#                     └──> simft train (needs both) ──> eval + sim-eval[0-2]
#   report (after every eval ends, ok or not)
#
# afterok edges with --kill-on-invalid-dep: a failed job cancels what depends
# on it instead of leaving it pending forever; the report still runs.
#
# Knobs (environment):
#   SEEDS="4551 4552 4553"           protocol §11 needs >= 3 per arm
#   ARMS="grounded residual simft"   simft needs residual for the same seeds
#   RUN_ROOT=$STORAGE_ROOT/runs/lewm-depth   next to the original LeWM run
#   TRAIN_GPUS=4                     GPUs per training job (DDP ranks)
#   GPU_SELECT=slurm                 see jepa_baseline.sbatch; "freest" races
#                                    when jobs start together
#   REFERENCE_EVALS=...              extra eval JSONs for the report (default:
#                                    the original lewm-depth run's, if present)
#   DRY_RUN=1                        print the sbatch commands, submit nothing
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/quad-wm}"
PROJECT_PARENT="$(dirname "$PROJECT_DIR")"
STORAGE_ROOT="${QUADWM_STORAGE_ROOT:-$PROJECT_PARENT/quad-wm-storage}"
SEEDS="${SEEDS:-4551 4552 4553}"
ARMS="${ARMS:-grounded residual simft}"
RUN_ROOT="${RUN_ROOT:-$STORAGE_ROOT/runs/lewm-depth}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"
DRY_RUN="${DRY_RUN:-0}"
export PROJECT_DIR QUADWM_STORAGE_ROOT="$STORAGE_ROOT" RUN_ROOT
export GPU_SELECT="${GPU_SELECT:-slurm}"
REFERENCE_EVALS="${REFERENCE_EVALS:-$(ls "$RUN_ROOT"/lewm-depth/eval/last-protocol-v1.json \
  "$RUN_ROOT"/lewm-depth/eval/last-sim-protocol-v2*.json 2>/dev/null | tr '\n' ' ' || true)}"

declare -A CONFIGS=(
  [grounded]=configs/jepa-wm/lewm_depth_grounded.yaml
  [residual]=configs/jepa-wm/lewm_depth_residual.yaml
  [simft]=configs/jepa-wm/lewm_depth_residual_simft.yaml
)
declare -A RUN_PREFIX=(
  [grounded]=lewm-depth-grounded
  [residual]=lewm-depth-residual
  [simft]=lewm-depth-residual-simft
)

cd "$PROJECT_DIR"
mkdir -p logs
for arm in $ARMS; do
  [[ -n "${CONFIGS[$arm]:-}" ]] || { echo "unknown arm $arm; choose from ${!CONFIGS[*]}" >&2; exit 1; }
done
if [[ " $ARMS " == *" simft "* && " $ARMS " != *" residual "* ]]; then
  for seed in $SEEDS; do
    parent="$RUN_ROOT/lewm-depth-residual-s$seed/last.pt"
    [[ -f "$parent" ]] || { echo "simft needs $parent (or add residual to ARMS)" >&2; exit 1; }
  done
fi

# Dry runs number fake jobs through a file: submit runs in a $(...) subshell.
FAKE_IDS="$(mktemp)"
trap 'rm -f "$FAKE_IDS"' EXIT
# submit <env assignments...> -- <sbatch args...>: prints the job id.
submit() {
  local env=()
  while [[ "$1" != "--" ]]; do env+=("$1"); shift; done
  shift
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "${env[*]} sbatch $*" >&2
    echo x >> "$FAKE_IDS"
    echo $((1000 + $(wc -l < "$FAKE_IDS")))
    return
  fi
  env "${env[@]}" sbatch --parsable --kill-on-invalid-dep=yes "$@" | cut -d';' -f1
}

after() {  # after <type> <ids...> -> --dependency flag, or nothing for no ids
  local type="$1"; shift
  if (( $# )); then echo "--dependency=$type:$(IFS=:; echo "$*")"; fi
}

GIT_COMMIT="$(git rev-parse HEAD)"
echo "pipeline: commit=$GIT_COMMIT seeds=($SEEDS) arms=($ARMS) run_root=$RUN_ROOT train_gpus=$TRAIN_GPUS gpu_select=$GPU_SELECT"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || echo "warning: uncommitted changes; runs record $GIT_COMMIT" >&2

# Fine-tuning episodes + EV1-sim episodes; a no-op when both are cached.
collect_id="$(submit -- scripts/slurm/jepa_sim_collect.sbatch)"
echo "sim-collect: $collect_id"

eval_ids=()
run_names=()
for seed in $SEEDS; do
  declare -A train_id=()
  for arm in $ARMS; do
    config="${CONFIGS[$arm]}"
    name="${RUN_PREFIX[$arm]}-s$seed"
    run_names+=("$name")
    deps=()
    if [[ "$arm" == simft ]]; then
      [[ -n "${train_id[residual]:-}" ]] && deps+=("${train_id[residual]}")
      deps+=("$collect_id")
    fi
    train_id[$arm]="$(submit CONFIG="$config" QUADWM_SEED="$seed" -- \
      --job-name="train-$name" --gres="gpu:$TRAIN_GPUS" $(after afterok "${deps[@]}") \
      scripts/slurm/jepa_baseline.sbatch)"
    real_id="$(submit CONFIG="$config" QUADWM_SEED="$seed" -- \
      --job-name="eval-$name" $(after afterok "${train_id[$arm]}") scripts/slurm/jepa_eval.sbatch)"
    # Sim shards also wait for sim-collect, so they never render EV1 episodes concurrently.
    sim_id="$(submit CONFIG="$config" QUADWM_SEED="$seed" -- \
      --job-name="sim-eval-$name" --array=0-2 $(after afterok "${train_id[$arm]}" $collect_id) \
      scripts/slurm/jepa_sim_eval.sbatch)"
    eval_ids+=("$real_id" "$sim_id")
    echo "$name: train=${train_id[$arm]} eval=$real_id sim-eval=$sim_id"
  done
  unset train_id
done

report_id="$(submit RUN_NAMES="${run_names[*]}" REFERENCE_EVALS="$REFERENCE_EVALS" -- \
  --job-name=report-lewm-pipeline $(after afterany "${eval_ids[@]}") scripts/slurm/jepa_report.sbatch)"
echo "report: $report_id -> $RUN_ROOT/reports/report-$report_id.md"
echo "watch: squeue -u \$USER -o '%.10i %.30j %.8T %.20E'"
