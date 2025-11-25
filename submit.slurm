#!/bin/bash
#SBATCH --job-name=age-train           # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --partition=gpu-beast          # GPU partition (gpu-standard or gpu-beast)
# SBATCH --gres=gpu:2                   # Request 2 GPUs (adjust as needed)
#SBATCH --output=%x-%j.log             # Output file jobname-jobid.log
#SBATCH --mail-type=ALL                # Send email for queue,start,error,completed etc...
#SBATCH --mail-user=rb3434w@gre.ac.uk  # Update to your email address

set -euo pipefail

module load www-proxy    # Load www-proxy for internet access (seemed required)
module load CUDA         # Load CUDA module

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
            OUTPUT_DIR="/home/rb3434w/CNN-PointNet-age-inference-age-inference/runs/${FOLDER_NAME}"

            echo ">>> Starting training (${MODALITY}${PTS:+, pts=${PTS}}, rgb=${RGB_BB}) -> ${OUTPUT_DIR}"
            python train_age.py \
                --data-root "/home/rb3434w/HandsDatasets" \
                --output-dir "${OUTPUT_DIR}" \
                --batch-size "${BATCH_SIZE}" \
                --epochs 40 \
                --seed "${SEED}" \
                --lr 0.0003 \
                --patience "${PATIENCE}" \
                --no-stratified-user-split \
                --loss-weight-nll "${LOSS_WEIGHT_NLL}" \
                --loss-weight-mse "${LOSS_WEIGHT_MSE}" \
                --loss-weight-mae "${LOSS_WEIGHT_MAE}" \
                --img-size "${IMG_SIZE}" \
                --rgb-backbone "${RGB_BB}" \
                "${EXTRA_FLAGS[@]}"
            echo ">>> Finished training (${MODALITY}${PTS:+, pts=${PTS}}, rgb=${RGB_BB})"
        done
    done
done
