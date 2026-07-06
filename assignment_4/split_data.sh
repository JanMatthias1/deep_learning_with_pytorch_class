#!/bin/bash
#SBATCH --job-name=detect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=75G
#SBATCH --time=25:00:00
#SBATCH --output=/users/jmatthia/deep_learning/code/assignment_4/split_data_%j.out

source /users/jmatthia/deep_learning/code/assignment_4/ucf50_env/bin/activate

cd /users/jmatthia/deep_learning/code/assignment_4

# --------- STEP 1: split dataset ----------
python data_split.py
# -----------------------------------------
    