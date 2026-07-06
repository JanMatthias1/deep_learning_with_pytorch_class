#!/bin/bash
# Setup script for DQN Assignment 6
# Usage: bash setup_env.sh
set -e

ENV=/users/jmatthia/deep_learning/env

echo "=== Creating conda environment ==="
conda create -p $ENV python=3.10 -y

echo "=== Installing PyTorch (CUDA 11.8) ==="
$ENV/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing dependencies from requirements.txt ==="
$ENV/bin/pip install -r dqn/requirements.txt

echo "=== Verifying installation ==="
$ENV/bin/python dqn/verify_env.py

echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV"
