#!/bin/bash
#SBATCH --job-name=swin_b_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/swin_b_test_%j.out
#SBATCH --error=logs/swin_b_test_%j.err


# Activate environment
source activate /users/jmatthia/deep_learning/env

# Run evaluation
cd /users/jmatthia/deep_learning/code/assignment_5
python test.py --checkpoint checkpoints/best_model.pt --batch-size 64 --data-root /users/jmatthia/deep_learning/data

