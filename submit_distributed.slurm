#!/bin/bash
#SBATCH --job-name=age-ddp               # Descriptive job name
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

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-PointNet-age-inference-age-inference}"
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
LOSS_WEIGHT_NLL="${LOSS_WEIGHT_NLL:-0.5}"
LOSS_WEIGHT_MSE="${LOSS_WEIGHT_MSE:-0.25}"
LOSS_WEIGHT_MAE="${LOSS_WEIGHT_MAE:-0.25}"
PATIENCE="${PATIENCE:-20}"
IMG_SIZE="${IMG_SIZE:-224}"
# RGB backbone options: restrict to efficientnet_b0/b2/b4 for this sweep
RGB_BACKBONES="${RGB_BACKBONES:-efficientnet_b0 efficientnet_b2 efficientnet_b4}"
# Modalities: rgb | pc | fused
MODALITIES="${MODALITIES:-rgb pc fused}"
# Point counts to try when point clouds are enabled
POINT_SIZES="${POINT_SIZES:-1024 2048 4096}"

for RGB_BB in ${RGB_BACKBONES}; do
  for MODALITY in ${MODALITIES}; do
    if [[ "${MODALITY}" == "rgb" ]]; then
      POINT_LIST=("none")
    else
      IFS=' ' read -r -a POINT_LIST <<< "${POINT_SIZES}"
    fi

    for PTS in "${POINT_LIST[@]}"; do
      EXTRA_FLAGS=()
      FOLDER_TAG="mod-${MODALITY}"

      if [[ "${MODALITY}" == "pc" ]]; then
        EXTRA_FLAGS+=(--use-pointcloud --no-rgb --num-points "${PTS}")
        FOLDER_TAG+="_pts${PTS}"
      elif [[ "${MODALITY}" == "fused" ]]; then
        EXTRA_FLAGS+=(--use-pointcloud --num-points "${PTS}")
        FOLDER_TAG+="_pts${PTS}"
      fi

      FOLDER_NAME="${FOLDER_TAG}_rgb${RGB_BB}_bs${BATCH_SIZE}_seed${SEED}_nll${LOSS_WEIGHT_NLL}_mse${LOSS_WEIGHT_MSE}_mae${LOSS_WEIGHT_MAE}"
      RUN_OUT="${OUTPUT_ROOT}/${FOLDER_NAME}"

      echo ">>> Starting distributed training (${MODALITY}${PTS:+, pts=${PTS}}, rgb=${RGB_BB}) -> ${RUN_OUT}"
      torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${GPUS_PER_NODE}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        "${PROJECT_ROOT}/train_distributed.py" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${RUN_OUT}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs 240 \
        --lr 3e-4 \
        --num-workers 8 \
        --seed "${SEED}" \
        --patience "${PATIENCE}" \
        --no-stratified-user-split \
        --loss-weight-nll "${LOSS_WEIGHT_NLL}" \
        --loss-weight-mse "${LOSS_WEIGHT_MSE}" \
        --loss-weight-mae "${LOSS_WEIGHT_MAE}" \
        --img-size "${IMG_SIZE}" \
        --rgb-backbone "${RGB_BB}" \
        --dist-backend nccl \
        "${EXTRA_FLAGS[@]}"
      echo ">>> Training (${MODALITY}${PTS:+, pts=${PTS}}, rgb=${RGB_BB}) complete. Artifacts in ${RUN_OUT}"
    done
  done
done
