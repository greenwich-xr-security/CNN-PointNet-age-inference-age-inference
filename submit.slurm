#!/bin/bash
#SBATCH --job=xr-security            # Give the job a name
#SBATCH --nodes=1                    # Number of nodes (only 1 gpu node)
#SBATCH --partition=gpu-beast        # GPU partition gpu-standard or gpu-beast
# SBATCH --gres=gpu:2                # Remove space for gpu-standard
#SBATCH --output=%x-%j.log           # Output file jobname-jobid.log
#SBATCH --mail-type=ALL              # Send email for queue,start,error,completed etc...
#SBATCH --mail-user=rb3434w@gre.ac.uk   # Send to email address (please change)

module load www-proxy                # Load www-proxy for internet access (seemed required)
module load CUDA                     # Load CUDA module

python train_age.py \
    --data-root "/projects/greenwich-xr-security/HandsDatasets" \
    --output-dir "${SLURM_SUBMIT_DIR:-.}/runs/test" \
    --model b7 \
    --batch-size 16 \
    --epochs 40 \
    --lr 3e-4 \
    --seed 42
