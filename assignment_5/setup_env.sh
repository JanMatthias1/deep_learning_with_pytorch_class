#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=setup_env_%j.out


# Change enviroment path as needed --> this is where the conda enviroment will be created 
ENV_DIR="/users/jmatthia/deep_learning/env"

# Remove existing env if present
if [ -d "$ENV_DIR" ]; then
    echo "Removing existing environment at $ENV_DIR"
    rm -rf "$ENV_DIR"
fi

echo "Creating conda environment at $ENV_DIR"
conda create --prefix "$ENV_DIR" python=3.10 -y

echo "Activating environment"
source activate "$ENV_DIR"

echo "Installing PyTorch (CUDA 11.8 — adjust if your cluster differs)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing additional dependencies"
pip install \
    matplotlib \
    numpy \
    tqdm \
    scikit-learn \
    wandb

echo "Environment created at: $ENV_DIR"
echo "Activate with: conda activate $ENV_DIR"

