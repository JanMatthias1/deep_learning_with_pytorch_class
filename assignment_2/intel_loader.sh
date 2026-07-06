#!/bin/bash
#SBATCH --dependency=afterok:28610810 
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-171
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --job-name=ssl_pretrain
#SBATCH --output=/users/jmatthia/deep_learning/code/ssl_intel_loader_%j.out
#SBATCH --error=/users/jmatthia/deep_learning/code/ssl_intel_loader_%j.err

# shell + conda
source ~/.bashrc
conda activate /users/jmatthia/deep_learning/env/dl_env_2

# go to code dir (important for relative paths)
cd /users/jmatthia/deep_learning/code

# execute notebook non-interactively
jupyter nbconvert \
  --to notebook \
  --execute intel_loader.ipynb \
  --output intel_loader_executed.ipynb \
  --output-dir /users/jmatthia/deep_learning/code \
  --debug
