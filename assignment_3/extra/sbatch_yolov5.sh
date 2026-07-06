#!/bin/bash
#SBATCH --job-name=yolov5m_voc
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-171
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=70:00:00
#SBATCH --output=/users/jmatthia/deep_learning/code/assignment_3/yolo_v5_training_%j.out

# shell + conda
source ~/.bashrc
conda activate /users/jmatthia/deep_learning/env/dl_env_2

# ---- W&B setup ----
export WANDB_API_KEY=""
export WANDB_PROJECT=assignment3_yolo
export WANDB_MODE=online   # important
# --------------------

cd /users/jmatthia/deep_learning/code/assignment_3/yolov5

python train.py \ 
    --img 640 \ 
    --batch-size -1
    --epochs 400 \ 
    --patience 3 \ 
    --data /users/jmatthia/deep_learning/data/pascal_voc_2012/VOC2012/VOC_split/voc.yaml \ 
    --cfg models/yolov5m.yaml \ 
    --weights '' \ 
    --project runs/train \ 
    --name yolov5m_voc_scratch \ 
    --save-period 1
  
