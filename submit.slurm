#!/bin/bash
#SBATCH --job-name=xr-security-train    # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --partition=gpu-beast           # GPU partition (gpu-standard or gpu-beast)
#SBATCH --gres=gpu:2                    # Request 2 GPUs (adjust as needed)
#SBATCH --output=%x-%j.log              # Output file jobname-jobid.log
#SBATCH --mail-type=ALL                 # Send email for queue,start,error,completed etc...
#SBATCH --mail-user=rb3434w@gre.ac.uk   # Update to your email address

set -euo pipefail

module load www-proxy                   # Load www-proxy for internet access if required
module load CUDA                        # Load CUDA module

# --- User configuration ------------------------------------------------------
DATA_ROOT="/projects/greenwich-xr-security/HandsDatasets"
MODEL_VARIANT="b7"                      # Options: b0-b7
BATCH_SIZE=16
EPOCHS=40
LEARNING_RATE=3e-4
SEED=42

# Directory for training artifacts (saved model, plots, logs)
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/runs/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "Starting training job ${SLURM_JOB_ID}"
echo "Dataset root : ${DATA_ROOT}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Model variant: ${MODEL_VARIANT}, Batch size: ${BATCH_SIZE}, Epochs: ${EPOCHS}, LR: ${LEARNING_RATE}, Seed: ${SEED}"

python train_age.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL_VARIANT}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --seed "${SEED}"
