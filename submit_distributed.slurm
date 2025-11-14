#!/bin/bash
#SBATCH --job-name=age-infer-ddp          # Descriptive job name
#SBATCH --nodes=1                         # Single node on gpu-beast
#SBATCH --ntasks-per-node=1               # torchrun will launch local ranks
#SBATCH --partition=gpu-beast             # GPU partition
#SBATCH --cpus-per-task=16                # CPU cores available to each task
#SBATCH --output=logs/%x-%j.log           # Stdout/stderr log file
#SBATCH --mail-type=ALL                   # Email notifications
#SBATCH --mail-user=rb3434w@gre.ac.uk     # TODO: change to preferred email

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
DATA_ROOT="${DATA_ROOT:-/home/rb3434w/HandsDatasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs}"

mkdir -p "${OUTPUT_ROOT}" logs

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

# Slurm rendezvous configuration (single node -> hostname works)
MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${SLURM_JOB_NUM_NODES:-1}
# gpu-beast does not export SLURM_GPUS_ON_NODE by default; assume 8 unless overridden.
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${GPUS_PER_NODE_OVERRIDE:-8}}}

export MASTER_ADDR MASTER_PORT

BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-3e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ -n "${MODEL_LIST:-}" ]]; then
  MODEL_SELECTION="${MODEL_LIST}"
elif [[ -n "${MODEL:-}" ]]; then
  MODEL_SELECTION="${MODEL}"
else
  MODEL_SELECTION="b0 b1 b2 b3 b4 b5 b6 b7"
fi

MODEL_SELECTION="${MODEL_SELECTION//,/ }"
read -r -a MODEL_VARIANTS <<< "${MODEL_SELECTION}"
if [[ ${#MODEL_VARIANTS[@]} -eq 0 ]]; then
  echo "[submit_distributed] ERROR: no EfficientNet variants were specified." >&2
  exit 1
fi

declare -a RUN_DIRS=()
declare -a MODEL_LABELS=()

for CURRENT_MODEL in "${MODEL_VARIANTS[@]}"; do
  FOLDER_NAME="${CURRENT_MODEL}_bs${BATCH_SIZE}_seed${SEED}"
  RUN_OUT="${OUTPUT_ROOT}/${FOLDER_NAME}"
  echo ">>> Starting distributed age inference training for model ${CURRENT_MODEL} (output: ${RUN_OUT})"
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${PROJECT_ROOT}/train_distributed.py" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${RUN_OUT}" \
    --model "${CURRENT_MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --dist-backend nccl

  echo ">>> Training complete for ${CURRENT_MODEL}. Artifacts in ${RUN_OUT}"
  RUN_DIRS+=("${RUN_OUT}")
  MODEL_LABELS+=("${CURRENT_MODEL}")
done

SUMMARY_FILE="${OUTPUT_ROOT}/comparison_${SLURM_JOB_ID:-$$}.tsv"
COMPARISON_REPORT="${OUTPUT_ROOT}/comparison_${SLURM_JOB_ID:-$$}_summary.txt"
COMPARE_ARGS=()
for dir in "${RUN_DIRS[@]}"; do
  COMPARE_ARGS+=("--run-dir" "${dir}")
done
for label in "${MODEL_LABELS[@]}"; do
  COMPARE_ARGS+=("--label" "${label}")
done

python "${PROJECT_ROOT}/compare_runs.py" \
  --summary "${SUMMARY_FILE}" \
  --report "${COMPARISON_REPORT}" \
  "${COMPARE_ARGS[@]}"

echo ">>> Comparison report written to ${COMPARISON_REPORT}"
