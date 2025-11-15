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

MODEL="${MODEL:-b4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"

FOLDER_NAME="${MODEL}_bs${BATCH_SIZE}_seed${SEED}"
RUN_OUT="${OUTPUT_ROOT}/${FOLDER_NAME}"

echo ">>> Starting distributed age inference training (output: ${RUN_OUT})"
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "${PROJECT_ROOT}/train_distributed.py" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${RUN_OUT}" \
  --model "${MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs 60 \
  --lr 3e-4 \
  --num-workers 8 \
  --seed "${SEED}" \
  --no-stratified-user-split \
  --loss-weight-nll 0.5 \
  --loss-weight-mse 0.25 \
  --loss-weight-mae 0.25 \
  --dist-backend nccl

echo ">>> Training complete. Artifacts in ${RUN_OUT}"
