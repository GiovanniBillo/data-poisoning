#!/bin/bash
#SBATCH --job-name=data_poisoning_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus=2
#SBATCH --mem=128G
#SBATCH --partition=GPU
#SBATCH --account=dssc
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# === Environment setup ===
echo "Activating virtualenv..." >&2
source ~/dlprojenv/bin/activate

mkdir -p logs
LOGFILE="logs/env_${SLURM_JOB_ID}.log"
PARAMS_FILE="brew_params.txt"

# === Detect GPU count ===
if [ -z "$SLURM_GPUS" ]; then
  echo "[WARN] SLURM_GPUS is not set. Using nvidia-smi to count." >&2
  SLURM_GPUS=$(nvidia-smi -L | wc -l)
fi

# === Read parameters ===
BREW_ARGS=()
while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# || -z "$line" ]] && continue
    line="${line#[[:space:]]*[0-9]*[[:space:]]}"
    line="${line%"${line##*[![:space:]]}"}"
    BREW_ARGS+=("$line")
done < "$PARAMS_FILE"

if [[ ${#BREW_ARGS[@]} -lt 3 ]]; then
    echo "[WARN] Using defaults: --net=ResNet18 --dataset=CIFAR10 --optimization=conservative" >&2
    BREW_ARGS+=(
        "--dataset=CIFAR10"
        "--optimization=conservative"
        "--net=ResNet18"
    )
fi

if ! printf '%s\n' "${BREW_ARGS[@]}" | grep -q -- "--deterministic"; then
    BREW_ARGS+=("--deterministic=False")
fi

# === Thread settings ===
THREADS_PER_GPU=$(( SLURM_CPUS_PER_TASK / SLURM_GPUS ))
export OMP_NUM_THREADS=$THREADS_PER_GPU
export MKL_NUM_THREADS=$THREADS_PER_GPU
export OPENBLAS_NUM_THREADS=$THREADS_PER_GPU
export VECLIB_MAXIMUM_THREADS=$THREADS_PER_GPU

# === Log environment ===
echo "==== SLURM ENVIRONMENT ===="     > "$LOGFILE"
env | grep SLURM_                   >> "$LOGFILE"
echo ""                             >> "$LOGFILE"

echo "==== GPU STATUS ===="          >> "$LOGFILE"
nvidia-smi                          >> "$LOGFILE" 2>/dev/null || echo "nvidia-smi not available" >> "$LOGFILE"

echo "==== TORCHRUN CMD ===="        >> "$LOGFILE"
echo "torchrun --nproc_per_node=$SLURM_GPUS brew_poison.py ${BREW_ARGS[@]}" >> "$LOGFILE"

# === Launch ===
torchrun --nproc_per_node=$SLURM_GPUS \
         brew_poison.py "${BREW_ARGS[@]}" >> "logs/train_${SLURM_JOB_ID}.log" 2>&1

