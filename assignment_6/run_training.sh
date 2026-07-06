#!/bin/bash
#SBATCH --job-name=dqn_cartpole
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dqn_%j.out
#SBATCH --error=logs/dqn_%j.err

# Usage: sbatch run_training.sh
# Or locally: bash run_training.sh

set -e

mkdir -p logs

# Load environment
source activate /users/jmatthia/deep_learning/env

echo "=== Starting DQN Training ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU mode')"

# W&B config
export WANDB_API_KEY=""  # paste your key here
export WANDB_PROJECT="DQN-CartPole"
export WANDB_MODE=online

# Run training
python dqn/dqn.py \
    --episodes 1500 \
    --lr 1e-4 \
    --batch-size 64 \
    --buffer-size 10000 \
    --gamma 0.99 \
    --eps-start 1.0 \
    --eps-end 0.01 \
    --eps-decay 500 \
    --target-update 3 \
    --warmup 1000 \
    --eval-interval 20 \
    --eval-episodes 10 \
    --wandb-project "DQN-CartPole"
 

echo "=== Training Complete ==="
