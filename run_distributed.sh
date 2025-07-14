#!/bin/bash
#SBATCH --job-name=data_poisoning_dist
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1              # One process per node; torchrun handles GPU spawning
#SBATCH --cpus-per-task=48               # Adapt to your CPU count
#SBATCH --mem=128G
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
PARAMS_FILE="brew_params.txt"

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

# === Read parameters ===
BREW_ARGS=()
while IFS= read -r line; do
    # Skip comments, empty lines, and lines with just whitespace
    [[ "$line" =~ ^[[:space:]]*# || -z "$line" ]] && continue
    # Remove any line numbers or leading/trailing whitespace
    line="${line#[[:space:]]*[0-9]*[[:space:]]}"
    line="${line%"${line##*[![:space:]]}"}"
    BREW_ARGS+=("$line")
done < "$PARAMS_FILE"

# === Set defaults if needed ===
if [[ ${#BREW_ARGS[@]} -lt 3 ]]; then
    echo "[WARN] Using defaults: --net=ResNet18 --dataset=CIFAR10 --optimization=conservative" >&2
    BREW_ARGS+=(
        "--dataset=CIFAR10"
        "--optimization=conservative"
        "--net=ResNet18"
    )
fi

# Ensure deterministic flag exists (default False)
if ! printf '%s\n' "${BREW_ARGS[@]}" | grep -q -- "--deterministic"; then
    BREW_ARGS+=("--deterministic=False")
fi

# === Log the final arguments ===
echo "Using arguments:" >&2
printf '  %s\n' "${BREW_ARGS[@]}" >&2


# === set optimal number of threads && related stuff ===
THREADS_PER_GPU=$(( SLURM_CPUS_PER_TASK / SLURM_GPUS_ON_NODE ))
export OMP_NUM_THREADS=$THREADS_PER_GPU
export MKL_NUM_THREADS=$THREADS_PER_GPU
export OPENBLAS_NUM_THREADS=$THREADS_PER_GPU
export VECLIB_MAXIMUM_THREADS=$THREADS_PER_GPU

# can try playing around with these as well 
# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads

echo "=== Thread Configuration ===" >> "$LOGFILE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK" >> "$LOGFILE"
echo "GPUs per node: $SLURM_GPUS_ON_NODE">> "$LOGFILE"
echo "Threads per GPU: $THREADS_PER_GPU">> "$LOGFILE"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS">> "$LOGFILE"

# Torch-specific settings
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Helps with debugging

echo "Launching torchrun..." >&2 >> "$LOGFILE"
echo "Command to be executed: " >> "$LOGFILE" 
echo "torchrun \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  dist_brew_poison.py  ${BREW_ARGS[@]}" >> "logs/train_${SLURM_JOB_ID}.log" >> "$LOGFILE"
echo " --local_rank=$SLURM_LOCALID will be passed directly in the script"

torchrun \
  --nproc_per_node="$SLURM_GPUS_ON_NODE" \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --node_rank="$SLURM_NODEID" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  dist_brew_poison.py "${BREW_ARGS[@]}" >> "logs/train_${SLURM_JOB_ID}.log" 2>&1
# torchrun --nproc_per_node=2 dist_brew_poison.py "${BREW_ARGS[@]}" >> "logs/train_${SLURM_JOB_ID}.log" 2>&1 

