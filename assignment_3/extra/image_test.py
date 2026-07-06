# core
import os
import argparse
import copy
import pickle
import random
# numerical / data
import numpy as np
import pandas as pd

# torch
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
from torchvision.datasets import STL10, ImageFolder
from torchvision.transforms import ToTensor, functional as TF

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# misc
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay

import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
#print("device name:", torch.cuda.get_device_name(0))

import wandb
from ultralytics import YOLO
from IPython.display import Image, display
import glob
from pathlib import Path
import kagglehub, os, shutil
import yaml as pyyaml 
import xml.etree.ElementTree as ET

random.seed(42)


# ensure outputs go under yolov5/runs/detect/exp*
os.chdir("/users/jmatthia/deep_learning/code/assignment_3/yolov5")

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="/users/jmatthia/deep_learning/code/assignment_3/yolov5/runs/train/yolov5m_voc_coco_pretrained4/weights/best.pt",  # local checkpoint
    source="local"
)

test_dir = Path("/users/jmatthia/deep_learning/data/pascal_voc_2012/VOC2012/VOC_split/images/test")
imgs = sorted([*test_dir.glob("*.jpg"), *test_dir.glob("*.png")])

print("n images:", len(imgs), "example:", imgs[0])

results = model(imgs)   # list of paths
results.save(save_dir="runs/detect/voc_test_preds")        # runs/detect/exp*