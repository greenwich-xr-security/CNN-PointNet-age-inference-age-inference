#!/bin/bash
#SBATCH --job-name=xr-security-train    # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --partition=gpu-beast           # GPU partition (gpu-standard or gpu-beast)
# SBATCH --gres=gpu:2                    # Request 2 GPUs (adjust as needed)
#SBATCH --output=%x-%j.log              # Output file jobname-jobid.log
#SBATCH --mail-type=ALL                 # Send email for queue,start,error,completed etc...
#SBATCH --mail-user=rb3434w@gre.ac.uk   # Update to your email address

set -euo pipefail

module load www-proxy    # Load www-proxy for internet access (seemed required)
module load CUDA         # Load CUDA module

MODELS="${MODELS:-${MODEL:-b0 b1 b2 b3 b4 b5 b6}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
LOSS_WEIGHT_NLL="${LOSS_WEIGHT_NLL:-0.5}"
LOSS_WEIGHT_MSE="${LOSS_WEIGHT_MSE:-0.25}"
LOSS_WEIGHT_MAE="${LOSS_WEIGHT_MAE:-0.25}"
PATIENCE="${PATIENCE:-20}"

for MODEL_NAME in ${MODELS}; do
    FOLDER_NAME="${MODEL_NAME}_bs${BATCH_SIZE}_seed${SEED}_nll${LOSS_WEIGHT_NLL}_mse${LOSS_WEIGHT_MSE}_mae${LOSS_WEIGHT_MAE}"
    OUTPUT_DIR="/home/rb3434w/CNN-age-inference/runs/${FOLDER_NAME}"

    echo ">>> Starting training for model ${MODEL_NAME} (output: ${OUTPUT_DIR})"
    python train_age.py \
        --data-root "/home/rb3434w/HandsDatasets" \
        --output-dir "${OUTPUT_DIR}" \
        --model "${MODEL_NAME}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs 40 \
        --seed "${SEED}" \
        --lr 0.0003 \
        --patience "${PATIENCE}" \
        --no-stratified-user-split \
        --loss-weight-nll "${LOSS_WEIGHT_NLL}" \
        --loss-weight-mse "${LOSS_WEIGHT_MSE}" \
        --loss-weight-mae "${LOSS_WEIGHT_MAE}"
    echo ">>> Finished training for model ${MODEL_NAME}"
done
