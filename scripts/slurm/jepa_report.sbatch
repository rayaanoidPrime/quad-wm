#!/bin/bash
#SBATCH --job-name=quadwm-report
#SBATCH --output=logs/report-%j.out
#SBATCH --error=logs/report-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

# Merge the eval JSONs of the named runs into one `quadwm report` markdown.
# Runs whose evals are missing (a failed job) are reported as missing, not
# fatal, so a partial pipeline still produces a report. Example:
#   RUN_ROOT=$STORAGE_ROOT/runs/lewm-depth RUN_NAMES="lewm-depth-grounded-s4551 lewm-depth-grounded-s4552" \
#     sbatch scripts/slurm/jepa_report.sbatch
PROJECT_DIR="${PROJECT_DIR:-$HOME/quad-wm}"
PROJECT_PARENT="$(dirname "$PROJECT_DIR")"
STORAGE_ROOT="${QUADWM_STORAGE_ROOT:-$PROJECT_PARENT/quad-wm-storage}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
RUN_NAMES="${RUN_NAMES:?set RUN_NAMES (space-separated run names under RUN_ROOT)}"
# Extra eval JSONs to compare against (e.g. the original single-seed LeWM), space-separated.
REFERENCE_EVALS="${REFERENCE_EVALS:-}"
# configs/eval/protocol.yaml and configs/eval/sim.yaml `name`s.
PROTOCOL_NAME="${PROTOCOL_NAME:-protocol-v1}"
SIM_NAME="${SIM_NAME:-sim-protocol-v2}"
REPORT="${REPORT:-$RUN_ROOT/reports/report-$SLURM_JOB_ID.md}"

module purge
module load python/3.11

cd "$PROJECT_DIR"
mkdir -p logs "$(dirname "$REPORT")"

log() { echo "[$(date -Is)] $*"; }

log "stage=job status=started id=$SLURM_JOB_ID host=$(hostname)"
shopt -s nullglob
evals=()
for name in $RUN_NAMES; do
  found=("$RUN_ROOT/$name/eval/last-$PROTOCOL_NAME.json" "$RUN_ROOT/$name/eval/last-$SIM_NAME"*.json)
  present=()
  for path in "${found[@]}"; do [[ -f "$path" ]] && present+=("$path"); done
  log "run=$name evals=${#present[@]}"
  (( ${#present[@]} )) || log "warning=no eval JSONs for $name"
  evals+=("${present[@]}")
done
for path in $REFERENCE_EVALS; do
  if [[ -f "$path" ]]; then evals+=("$path"); else log "warning=missing reference $path"; fi
done
(( ${#evals[@]} )) || { log "error=no eval JSONs found"; exit 1; }

log "stage=report status=starting evals=${#evals[@]} output=$REPORT"
uv run quadwm report "${evals[@]}" --output "$REPORT" > /dev/null
log "stage=report status=complete output=$REPORT"
log "stage=job status=complete"
