#!/bin/bash
#SBATCH --dependency=afterok:29718208 
#SBATCH --job-name=detect
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=75G
#SBATCH --time=25:00:00
#SBATCH --output=/users/jmatthia/deep_learning/code/assignment_4/training_run_%j.out

source /users/jmatthia/deep_learning/code/assignment_4/ucf50_env/bin/activate

cd /users/jmatthia/deep_learning/code/assignment_4/video_rec


# ---- W&B setup ----
export WANDB_API_KEY=""
export WANDB_PROJECT=ucf50-action-recognition
export WANDB_MODE=online   # important
# --------------------

#run 1 --> without freezing the backbone, but with a smaller learning rate and more epochs 
python run.py \
    --train_dir /users/jmatthia/deep_learning/data/UCF50_splits/train \
    --val_dir   /users/jmatthia/deep_learning/data/UCF50_splits/val \
    --rnn_n_layers 1 \
    --learning_rate 3e-5 \
    --n_classes 50 \
    --batch_size 8 \
    --n_epochs 60 \
    --mode train

"""
python run.py \
    --train_dir /users/jmatthia/deep_learning/data/UCF50_splits/train \
    --val_dir   /users/jmatthia/deep_learning/data/UCF50_splits/val \
    --model_type lrcn \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 512 \
    --rnn_n_layers 2 \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 8 \
    --dropout 0.5 \
    --learning_rate 3e-5 \
    --n_epochs 60 \
    --mode train

# run 2 --> freezing the backbone for the first 10 epochs
python run.py \
    --train_dir /users/jmatthia/deep_learning/data/UCF50_splits/train \
    --val_dir   /users/jmatthia/deep_learning/data/UCF50_splits/val \
    --model_type lrcn \
    --cnn_backbone resnet34 \
    --rnn_hidden_size 256 \
    --rnn_n_layers 1 \
    --n_classes 50 \
    --batch_size 8 \
    --dropout 0.5 \
    --learning_rate 1e-4 \
    --n_epochs 40 \
    --mode train
    
python run.py \
    --train_dir /users/jmatthia/deep_learning/data/UCF50_splits/train \
    --val_dir   /users/jmatthia/deep_learning/data/UCF50_splits/val \
    --rnn_n_layers 1 \
    --learning_rate 3e-5 \
    --n_classes 50 \
    --batch_size 8 \
    --n_epochs 60 \
    --dropout 0.5 \
    --mode train

"""
