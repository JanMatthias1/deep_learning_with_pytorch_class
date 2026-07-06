#!/bin/bash
#SBATCH --job-name=3dinfomax_embed
#SBATCH --output=/home/jmatthi2/deep_learning/COMET/3dinfomax/logs/inference_%j.out
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# run_3dinfomax_inference.sh
# Extracts 3D Infomax embeddings for lipid SMILES.
#
# Prerequisites:
#   1. Run create_env_3dinfomax.sh first
#   2. Ensure lance_lipid_smiles.csv is in SCRIPT_DIR
#
# Usage:
#   sbatch run_3dinfomax_inference.sh

# ---- PATHS ----
ENV_PATH="/home/jmatthi2/deep_learning/env_3dinfomax"
SCRIPT_DIR="/home/jmatthi2/deep_learning/COMET/3dinfomax"
REPO_DIR="${SCRIPT_DIR}/3DInfomax"
CSV_PATH="${SCRIPT_DIR}/lance_lipid_smiles.csv"
OUTPUT_PATH="${SCRIPT_DIR}/3dinfomax_lipid_embeddings.npy"

module load anaconda3/2024.02-1
conda activate "${ENV_PATH}"

echo "Running 3D Infomax inference..."
echo "CSV:    ${CSV_PATH}"
echo "Repo:   ${REPO_DIR}"
echo "Output: ${OUTPUT_PATH}"
echo ""

python "${SCRIPT_DIR}/3dinfomax_inference.py" \
    --csv_path "${CSV_PATH}" \
    --repo_dir "${REPO_DIR}" \
    --output_path "${OUTPUT_PATH}"

echo ""
echo "Done."
