#!/bin/bash
#SBATCH --job-name=vae_train
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # torchrun handles multi-processing, so 1 task is fine
#SBATCH --cpus-per-task=4           # Adjust based on your cluster's CPU-to-GPU ratio
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=job_%j.out         # Standard output and error log (%j inserts Job ID)

module load conda
conda activate myenv
export TORCHINDUCTOR_CACHE_DIR=".inductor_cache"
python -u train_vae.py
