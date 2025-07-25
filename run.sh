#!/bin/bash
#SBATCH --job-name=data_poisoning_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --partition=GPU
#SBATCH --account=dssc
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# === Environment setup ===
echo "Activating virtualenv..." >&2
source ~/dlprojenv/bin/activate
# === Environment setup ===
VENV_DIR=~/dlprojenv

echo "Checking Python virtual environment..." >&2
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[INFO] Virtualenv not found. Creating new environment at $VENV_DIR..." >&2
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "[INFO] Installing requirements..." >&2
    pip install --upgrade pip
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    else
        echo "[WARN] No requirements.txt found. Skipping dependency install." >&2
    fi
else
    echo "[INFO] Activating existing virtualenv..." >&2
    source "$VENV_DIR/bin/activate"
fi


mkdir -p logs
TIMESTAMP=$(date "+%Y%m%dS")
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

# === Inject per-job poisonkey and eps ===
GRID_FILE="param_grid.csv"
TASK_INDEX=$((SLURM_ARRAY_TASK_ID + 1))  # skip header

IFS=, read -r eps poisonkey target goal < <(tail -n +$((TASK_INDEX + 1)) "$GRID_FILE" | head -n1)

echo "[INFO] Running SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> eps=$eps, poisonkey=$poisonkey (target=$target, goal=$goal)" >&2

# Inject into args
BREW_ARGS+=("--eps=$eps")
BREW_ARGS+=("--poisonkey=$poisonkey")

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
         brew_and_visualize_poison2.py "${BREW_ARGS[@]}" >> "logs/train_${SLURM_JOB_ID}.log" 2>&1

