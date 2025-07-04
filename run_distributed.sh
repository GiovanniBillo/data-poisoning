#!/bin/bash
#SBATCH --job-name=torch_distrib
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1              # One process per node; torchrun handles GPU spawning
#SBATCH --cpus-per-task=48               # Adapt to your CPU count
#SBATCH --gpus-per-node=2                # Important: use this (not --gpus=V100:2)
#SBATCH --partition=GPU
#SBATCH --account=dssc
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%j.out       # Unified output
#SBATCH --error=logs/train_%j.err

# === Environment prep ===
echo "Activating virtualenv..." >&2
source ~/dlprojenv/bin/activate

# === Create logs directory if missing ===
mkdir -p logs

# === Set master address and port ===
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=11234

# === Determine number of GPUs on this node ===
if [ -z "$SLURM_GPUS_ON_NODE" ]; then
  echo "[WARN] SLURM_GPUS_ON_NODE is not set. Falling back to nvidia-smi." >&2
  SLURM_GPUS_ON_NODE=$(nvidia-smi -L | wc -l)
fi

# === Log useful environment info ===
LOGFILE="logs/env_${SLURM_JOB_ID}.log"
echo "==== SLURM ENVIRONMENT ===="     > "$LOGFILE"
env | grep SLURM_                   >> "$LOGFILE"
echo ""                             >> "$LOGFILE"

echo "==== CUDA + GPU ENV ===="      >> "$LOGFILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "$LOGFILE"
nvidia-smi                          >> "$LOGFILE" 2>/dev/null || echo "nvidia-smi not available" >> "$LOGFILE"

echo ""                             >> "$LOGFILE"
echo "==== PYTORCHRUN LAUNCH ===="  >> "$LOGFILE"
echo "MASTER_ADDR=$MASTER_ADDR"     >> "$LOGFILE"
echo "MASTER_PORT=$MASTER_PORT"     >> "$LOGFILE"
echo "NODE_RANK=$SLURM_NODEID"      >> "$LOGFILE"
echo "NPROC_PER_NODE=$SLURM_GPUS_ON_NODE" >> "$LOGFILE"
echo "NNODES=$SLURM_JOB_NUM_NODES"  >> "$LOGFILE"
echo ""                             >> "$LOGFILE"

# === Launch distributed training ===
echo "Launching torchrun..." >&2
torchrun \
  --nproc_per_node="$SLURM_GPUS_ON_NODE" \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --node_rank="$SLURM_NODEID" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  dist_brew_poison.py >> "logs/train_${SLURM_JOB_ID}.log" 2>&1

