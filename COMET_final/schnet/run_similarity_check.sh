#!/bin/bash
#SBATCH --job-name=schnet_simcheck
#SBATCH --output=logs/schnet_simcheck_%j.out
#SBATCH --error=logs/schnet_simcheck_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

ENV_PATH="/home/jmatthi2/deep_learning/env_schnet"
SCRIPT_DIR="/home/jmatthi2/deep_learning/COMET/schnet"
EMBEDDINGS_PATH="${SCRIPT_DIR}/schnet_lipid_embeddings.npy"
NAMES_CSV="${SCRIPT_DIR}/lance_lipid_smiles.csv"
OUTPUT_PLOT="${SCRIPT_DIR}/schnet_similarity_matrix.png"

mkdir -p logs

module load anaconda3/2024.02-1
conda activate "${ENV_PATH}"

python "${SCRIPT_DIR}/similarity_check_schnet.py" \
    --embeddings_path "${EMBEDDINGS_PATH}" \
    --names_csv "${NAMES_CSV}" \
    --output_path "${OUTPUT_PLOT}"
