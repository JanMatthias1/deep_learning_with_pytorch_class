#!/bin/bash
#SBATCH --output=logs/create_env_schnet_%j.out
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

ENV_PATH="/home/jmatthi2/deep_learning/env_schnet"
PYTHON_VERSION="3.10"

module load anaconda3/2024.02-1

echo "============================================"
echo "Creating SchNet environment at: ${ENV_PATH}"
echo "============================================"

conda create --prefix "${ENV_PATH}" python=${PYTHON_VERSION} -y
conda activate "${ENV_PATH}"

conda install pytorch cpuonly -c pytorch -y
pip install torch_geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install schnetpack==1.0.1
conda install -c conda-forge rdkit=2023.09 -y
pip install pandas matplotlib numpy

echo "============================================"
echo "Environment created at: ${ENV_PATH}"
echo "============================================"