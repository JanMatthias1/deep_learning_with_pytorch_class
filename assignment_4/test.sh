#!/bin/bash
#SBATCH --job-name=detect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=75G
#SBATCH --time=25:00:00
#SBATCH --output=/users/jmatthia/deep_learning/code/assignment_4/test_best_mode_%j.out


source /users/jmatthia/deep_learning/code/assignment_4/ucf50_env/bin/activate
cd /users/jmatthia/deep_learning/code/assignment_4/video_rec

# add testing line here 
python run.py \
    --ckpt /users/jmatthia/deep_learning/code/assignment_4/video_rec/models/best_model_wts.pt \
    --test_dir /users/jmatthia/deep_learning/data/UCF50_splits/test \
    --model_type lrcn \
    --n_classes 50 \
    --batch_size 4 \
    --mode eval \
    --output_path /users/jmatthia/deep_learning/code/assignment_4/
