#!/bin/bash
#SBATCH --job-name=3dinfomax_simcheck
#SBATCH --output=/home/jmatthi2/deep_learning/COMET/3dinfomax/logs/simcheck_%j.out
#SBATCH --error=/home/jmatthi2/deep_learning/COMET/3dinfomax/logs/simcheck_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# run_simcheck_3dinfomax.sh
# Runs pairwise cosine similarity check on 3D Infomax embeddings.
#
# Prerequisites:
#   1. run_3dinfomax_inference.sh completed successfully
#
# Usage:
#   sbatch run_simcheck_3dinfomax.sh

# ---- PATHS ----
ENV_PATH="/home/jmatthi2/deep_learning/env_3dinfomax"
SCRIPT_DIR="/home/jmatthi2/deep_learning/COMET/3dinfomax"
EMBEDDINGS_PATH="${SCRIPT_DIR}/3dinfomax_lipid_embeddings.npy"
NAMES_CSV="${SCRIPT_DIR}/lance_lipid_smiles.csv"
OUTPUT_PLOT="${SCRIPT_DIR}/3dinfomax_similarity_matrix.png"

module load anaconda3/2024.02-1
conda activate "${ENV_PATH}"

echo "Running 3D Infomax similarity check..."
echo "Embeddings: ${EMBEDDINGS_PATH}"
echo ""

python "${SCRIPT_DIR}/similarity_check_3dinfomax.py" \
    --embeddings_path "${EMBEDDINGS_PATH}" \
    --names_csv "${NAMES_CSV}" \
    --output_path "${OUTPUT_PLOT}"

echo ""
echo "Done."
