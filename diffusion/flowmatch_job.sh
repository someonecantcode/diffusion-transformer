#!/bin/bash
#SBATCH --job-name=flowmatch
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --gpus=4
#SBATCH --constraint="a40|l40|l40s|a100|h200"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # torchrun handles multi-processing, so 1 task is fine
#SBATCH --cpus-per-gpu=4           # Adjust based on your cluster's CPU-to-GPU ratio
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=job_%j.out         # Standard output and error log (%j inserts Job ID)

module load conda
conda activate myenv
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR=".inductor_cache"
torchrun --standalone --nproc_per_node=4 train.py