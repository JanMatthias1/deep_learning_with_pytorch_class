# Swin-B on CIFAR-100 (from scratch)

Train and evaluate a Swin-Base transformer on CIFAR-100 using a SLURM cluster with GPU.

## Project structure

```
├── train.py          # Training loop (Swin-B, AdamW, cosine LR, early stopping, W&B logging)
├── test.py           # Evaluation: accuracy, per-class precision/recall/F1, confusion matrix, ROC-AUC
├── data.py           # CIFAR-100 loading, 90/10 train/val split, augmentation (RandAugment + resize to 224×224)
├── setup_env.sh      # SLURM job: creates conda env and installs all dependencies (change env path as needed)
├── run_train.sh      # SLURM job: launches training
├── run_test.sh       # SLURM job: launches evaluation on test set
├── checkpoints/      # Saved model weights (best_model.pt)
├── logs/             # SLURM stdout/stderr logs
└── results/          # Evaluation outputs (confusion_matrix.png)
```

## Environment setup

Requires: conda, CUDA 11.8 compatible GPU node.

**Option A — via SLURM:**
```bash
sbatch setup_env.sh
```
This creates a conda environment at `/users/jmatthia/deep_learning/env` with Python 3.10.

### Dependencies

| Package | Purpose |
|---|---|
| torch, torchvision, torchaudio | Model, data loading, transforms |
| wandb | Experiment tracking |
| matplotlib | Confusion matrix plot |
| scikit-learn | Classification report, ROC-AUC |
| numpy | Array operations |


## Training

1. Edit `run_train.sh` to set your data path and W&B API key. 
2. Submit:
```bash
sbatch run_train.sh
```

This runs `train.py` with the following defaults:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Max training epochs |
| `--batch-size` | 64 | Batch size per GPU |
| `--lr` | 1e-3 | Initial learning rate |
| `--weight_decay` | 0.05 | AdamW weight decay |
| `--patience` | 10 | Early stopping patience (epochs without val improvement) |
| `--data-root` | `data` | Path to CIFAR-100 download directory |

The best checkpoint (by validation accuracy) is saved to `checkpoints/best_model.pt`. 

## Evaluation

```bash
sbatch run_test.sh
```

This runs `test.py`, which loads `checkpoints/best_model.pt` and reports on the held-out test set:

- Overall accuracy
- Per-class precision, recall, F1 (via `classification_report`)
- Confusion matrix (saved to `results/confusion_matrix.png`)
- Macro-averaged ROC-AUC (one-vs-rest)

### Manual evaluation

This automatically creates the result folder.

```bash
python test.py --checkpoint checkpoints/best_model.pt --data-root /path/to/data --batch-size 64
```

## Script details

**`data.py`** — Downloads CIFAR-100, splits the 50k training set into 45k train / 5k validation (fixed seed=42). Training images are resized to 224×224 and augmented with random crops, horizontal flips, and RandAugment. Validation and test images are resized and normalized only.

**`train.py`** — Builds a Swin-Base (87M params) with a 100-class head initialized from scratch. Optimized with AdamW and cosine annealing LR schedule. Checkpoints the best model by validation accuracy and stops early if no improvement for `--patience` epochs.

**`test.py`** — Loads a trained checkpoint, runs inference on the CIFAR-100 test set, and computes classification metrics. Outputs a confusion matrix heatmap and prints a full classification report.
