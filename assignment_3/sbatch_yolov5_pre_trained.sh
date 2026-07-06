#!/bin/bash
#SBATCH --job-name=yolov5m_voc
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:tesa100:1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=75G
#SBATCH --time=10:00:00
#SBATCH --output=/users/jmatthia/deep_learning/code/assignment_3/yolo_v5_training_%j.out

# shell + conda
source ~/.bashrc
conda activate /users/jmatthia/deep_learning/env/yolov5_cu121

which python
python -c "import sys; print(sys.executable)"

# print GPU info
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
    print("CUDA version (torch):", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
PY

# ---- W&B setup ----
export WANDB_API_KEY=""
export WANDB_PROJECT=assignment3_yolo
export WANDB_MODE=online   # important
# --------------------

cd /users/jmatthia/deep_learning/code/assignment_3/yolov5

python train.py \
  --img 640 \
  --batch-size -1\
  --epochs 200 \
  --patience 20 \
  --data /users/jmatthia/deep_learning/data/pascal_voc_2012/VOC2012/VOC_split/voc.yaml \
  --cfg models/yolov5m.yaml \
  --weights yolov5m.pt \
  --project runs/train \
  --name yolov5m_voc_coco_pretrained \
  --save-period 25
  
