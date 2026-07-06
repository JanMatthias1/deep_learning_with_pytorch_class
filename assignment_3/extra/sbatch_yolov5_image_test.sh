#!/bin/bash
#SBATCH --job-name=yolo5m_test
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:tesa100:1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=5:00:00
#SBATCH --output=/users/jmatthia/deep_learning/code/assignment_3/yolo_v5_test_%j.out

# shell + conda
source ~/.bashrc
conda activate /users/jmatthia/deep_learning/env/yolov5_cu121 

# run script
python /users/jmatthia/deep_learning/code/assignment_3/image_test.py