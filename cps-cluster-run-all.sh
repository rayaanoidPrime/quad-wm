#!/bin/bash
#SBATCH --job-name=uako-all
#SBATCH --output=logs/uako_all_%j.log
#SBATCH --error=logs/uako_all_%j.err
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=25GB
#SBATCH --partition=debug
#SBATCH --gres=gpu:4

# =============================================================================
# Train ALL systems in parallel, each on its own GPU.
#
# Requests 4 GPUs and pins each system to a separate device via
# CUDA_VISIBLE_DEVICES, avoiding shared-context overhead.
#
# To fall back to shared single-GPU mode (e.g., if 4 GPUs are unavailable):
#   sbatch --gres=gpu:1 scripts/cps-cluster-run-all.sh
# Systems will round-robin over whatever GPUs are allocated.
#
# Usage:
#   sbatch scripts/cps-cluster-run-all.sh
#   sbatch scripts/cps-cluster-run-all.sh 42            # custom seed
# =============================================================================

set -euo pipefail

SEED="${1:-4551}"

# --- Paths ---
PROJECT_DIR="$HOME/robotics/universal-koopman"
VENV_DIR="$PROJECT_DIR/.venv"

# --- Modules ---
module purge
module load python/3.11
module load rocm/6.2.4

# --- Environment ---
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
mkdir -p logs

# --- Print info ---
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Training ALL systems in parallel"
echo "Seed:          $SEED"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')
echo "GPUs:          $NUM_GPUS"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  GPU $i:       $(python -c "import torch; print(torch.cuda.get_device_name($i))" 2>/dev/null || echo 'N/A')"
done
echo "============================================"

SYSTEMS="pendulum cartpole two_link planar_quadrotor quad3d"    
PIDS=()

GPU_IDX=0
for sys in $SYSTEMS; do
    GPU=$((GPU_IDX % NUM_GPUS))
    echo "[$(date +%H:%M:%S)] Starting $sys on GPU $GPU ..."
    CUDA_VISIBLE_DEVICES=$GPU python -m universal_koopman.train \
        --system "$sys" \
        --seed "$SEED" \
        > "logs/uako_${SLURM_JOB_ID}_${sys}.log" \
        2> "logs/uako_${SLURM_JOB_ID}_${sys}.err" &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
    GPU_IDX=$((GPU_IDX + 1))
done

echo ""
echo "All ${#PIDS[@]} systems launched. Waiting..."
echo "============================================"

# --- Wait and report per-system exit status ---
FAILED=0
i=0
for sys in $SYSTEMS; do
    wait "${PIDS[$i]}" && \
        echo "[$(date +%H:%M:%S)] $sys finished successfully." || \
        { echo "[$(date +%H:%M:%S)] $sys FAILED (exit code $?)."; FAILED=$((FAILED+1)); }
    i=$((i+1))
done

echo ""
echo "============================================"
echo "All done at $(date). Failures: $FAILED/${#PIDS[@]}"
echo "============================================"

exit $FAILED
