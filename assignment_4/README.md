# Video Classification with UCF50

This project implements a video action recognition pipeline on the [UCF50 dataset](https://www.crcv.ucf.edu/data/UCF50.rar) using an LRCN (Long-term Recurrent Convolutional Network). Spatial features are extracted per frame via a pretrained ResNet backbone; temporal dynamics across frames are modeled by an LSTM. The pipeline covers frame extraction, group-aware data splitting, training with learning rate scheduling, and full evaluation with classification reports, ROC AUC, F1, and confusion matrix.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Preprocessing, Frame Extraction, and Data Split](#preprocessing-frame-extraction-and-data-split)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hyperparameters Reference](#hyperparameters-reference)
- [Results](#results)
- [Training Logs](#training-logs)

---

## Project Structure

```
assignment_4/
├── video_rec/
│   ├── models.py            # LRCN model definition (ResNet backbone + LSTM)
│   ├── train.py             # Training loop with W&B logging and LR scheduling
│   ├── test.py              # Evaluation: accuracy, F1, AUC, confusion matrix
│   ├── video_datasets.py    # VideoDataset, load_dataset, dataset_split, collate functions
│   ├── utils.py             # Frame extraction, transforms, DataLoader factories
│   ├── run.py               # Main entry point for train and eval modes
│   └── run_training.py      # Standalone training pipeline (alternative entry point)
├── data_split.py            # Frame extraction + group-aware train/val/test split
├── train.sh                 # SLURM script: training job
├── test.sh                  # SLURM script: evaluation job
├── split_data.sh            # SLURM script: preprocessing job
├── setup_env.sh             # Environment setup
├── requirements.txt         # Python dependencies

```

## Dataset Preparation
 
Download the UCF50 dataset from [https://www.crcv.ucf.edu/data/UCF50.rar](https://www.crcv.ucf.edu/data/UCF50.rar). It contains videos for 50 human action classes.
 
Unpack it so the structure is:
 
```
UCF50/
├── BaseballPitch/
├── Basketball/
├── ...
└── YoYo/
```

---

## Environment Setup
 
Run the setup script to create a virtual environment (`ucf50_env`) and install all dependencies:
 
```bash
bash setup_env.sh
```
This installs packages from `requirements.txt`, including `torch>=2.0`, `torchvision>=0.15`, `opencv-python`, `scikit-learn`, `seaborn`, `tqdm`, and `Pillow`.

## Hardware Requirements

A CUDA-enabled GPU is recommended for training. The code automatically detects GPU availability.

---

## Preprocessing, Frame Extraction, and Data Split
 
Raw videos must be converted to frame sequences before training. The script also creates group-aware splits so that videos from the same recording group (identified by `_gXX_` in filenames) never appear in more than one split, preventing data leakage.
 
```bash
bash split_data.sh
```
 
This runs `data_split.py`, which:
 
- Extracts **16 uniformly sampled frames** per video and saves them as JPEG images
- Splits data into **train / val / test** using group-aware stratified sampling (default: 70% / 20% / 10%)
- Outputs frame directories to `/users/jmatthia/deep_learning/data/UCF50_splits/` with `train/`, `val/`, and `test/` subdirectories. Update the output path if needed.
 
Each output subdirectory follows this structure:
 
```
UCF50_splits/
├── train/
│   ├── BaseballPitch/
│   │   ├── v_BaseballPitch_g01_c01/
│   │   │   ├── frame0.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── val/
└── test/
```

---

## Training the Model

### Step 1: Run Training

Update `--train_dir` and `--val_dir` in `train.sh` to point to your split directories, then submit:
 
```bash
bash train.sh
```
 
The training script (`run.py --mode train`) performs:
 
1. Loads train and val splits from the pre-split directories via `load_dataset`
2. Applies data augmentation for training (random horizontal flip, random affine) and standard resize + normalize for validation
3. Wraps data in `VideoDataset` with padded-sequence collation (`collate_fn_rnn`)
4. Initializes **LRCN**: ResNet backbone (default: `resnet18`) with the classification head replaced by `Identity`, feeding into a single-layer LSTM (hidden size 100), followed by dropout and a linear head for 50 classes
5. Trains with **Adam** optimizer (`lr=3e-5`), **CrossEntropyLoss** (sum reduction), and **ReduceLROnPlateau** scheduler (factor 0.5, patience 5)
6. Saves the best checkpoint by validation accuracy to `./models/best_model_wts.pt`
7. Logs train/val loss and accuracy to **Weights & Biases** (project: `ucf50-action-recognition`)

---

## Testing and Evaluation

## Evaluation
 
Update `--ckpt` in `test.sh` if needed (default points to `./models/best_model_wts.pt`), then submit:
 
```bash
bash test.sh
```
 
The evaluation script (`run.py --mode eval`) performs:
 
1. Loads the test split from `--test_dir`
2. Loads the model checkpoint specified by `--ckpt`
3. Runs inference and reports:
   - **Overall test accuracy**
   - **Macro ROC AUC** (one-vs-rest)
   - **Macro F1 score**
   - **Per-class classification report** (precision, recall, F1, support)
   - **Confusion matrix** saved to `--output_path/confusion_matrix.png`
 

## Results
 
Trained on UCF50 (50 action classes, 16 frames/video, ResNet18 backbone, 1-layer LSTM, 60 epochs).
 
| Metric | Value |
|---|---|
| Test Accuracy | **75.88%** |
| Macro F1 | **0.7439** |
| Macro ROC AUC | **0.9720** |
 
Confusion matrix saved to `confusion_matrix.png`.

## Training Logs
 
SLURM output logs for each stage of the pipeline:
 
| Stage | Log file |
|---|---|
| Preprocessing & data split | `split_data_29718208.out` |
| Training run | `training_run_29718415.out` |
| Evaluation | `test_best_mode_29748135.out` |
 
Logs are located in `/users/jmatthia/deep_learning/code/assignment_4/`.