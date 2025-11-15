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

MODEL="b4"
BATCH_SIZE=32
SEED=42

FOLDER_NAME="${MODEL}_bs${BATCH_SIZE}_seed${SEED}"
OUTPUT_DIR="/home/rb3434w/CNN-age-inference/runs/${FOLDER_NAME}"

python train_age.py \
    --data-root "/home/rb3434w/HandsDatasets" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs 40 \
    --seed "${SEED}" \
    --lr 0.0003 \
    --no-stratified-user-split
