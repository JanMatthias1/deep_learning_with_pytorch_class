#!/bin/bash
#SBATCH --job-name=schnet_embed
#SBATCH --output=logs/schnet_embed_%j.out
#SBATCH --error=logs/schnet_embed_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

ENV_PATH="/home/jmatthi2/deep_learning/env_schnet"
SCRIPT_DIR="/home/jmatthi2/deep_learning/COMET/schnet"
CSV_PATH="${SCRIPT_DIR}/lance_lipid_smiles.csv"
OUTPUT_PATH="${SCRIPT_DIR}/schnet_lipid_embeddings.npy"
QM9_DIR="${SCRIPT_DIR}/qm9_data"

mkdir -p logs

module load anaconda3/2024.02-1
conda activate "${ENV_PATH}"

python "${SCRIPT_DIR}/schnet_inference.py" \
    --csv_path "${CSV_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --qm9_data_dir "${QM9_DIR}" \
    --target 7 \
    --max_attempts 5000