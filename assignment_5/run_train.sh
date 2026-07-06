#!/bin/bash
#SBATCH --job-name=swin_b_cifar100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/swin_b_train_%j.out
#SBATCH --error=logs/swin_b_train_%j.err

# Create log directory
mkdir -p /users/jmatthia/deep_learning/code/assignment_5/logs

# Activate environment
source activate /users/jmatthia/deep_learning/env


# ---- W&B setup ----
export WANDB_API_KEY=""
export WANDB_PROJECT=CIFAR100-SwinB-Training
export WANDB_MODE=online   # important
# --------------------

# Run training
cd /users/jmatthia/deep_learning/code/assignment_5
python train.py --epochs 70 --batch-size 64 --lr 2e-4 --data-root /users/jmatthia/deep_learning/data
