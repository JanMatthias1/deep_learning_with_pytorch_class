# Assignment 2: Image Classification with Self-Supervised Rotation Prediction & Transfer Learning

This repository contains the informations for Assignment 2, which consists of:

1. Self-supervised pretraining using rotation prediction.
2. Supervised training and transfer learning on the Intel Images dataset.
3. Experiments across multiple seeds.
4. Cluster execution using L40S GPUs.

All hyperparameters and seed settings are defined directly inside the Jupyter notebooks. They are reproduced below to provide a complete global overview of the experimental setup.

All notebooks were executed on the cluster using NVIDIA L40S GPUs. The corresponding `sbatch` scripts used for submission are included below.

---

# Self-Supervised Learning (SLL) Pretrained Model

## Seed Configuration and Hyperparameter configuration 

The following seed configuration was used to ensure reproducibility:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

g = torch.Generator().manual_seed(SEED)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=3e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
epochs = 50
```

## Cluster Sbatch 

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-171
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --job-name=ssl_pretrain
#SBATCH --output=/users/jmatthia/deep_learning/code/ssl_pretrain_%j.out
#SBATCH --error=/users/jmatthia/deep_learning/code/ssl_pretrain_%j.err

# shell + conda
source ~/.bashrc
conda activate /users/jmatthia/deep_learning/env/dl_env_2

# go to code dir (important for relative paths)
cd /users/jmatthia/deep_learning/code

# execute notebook non-interactively
jupyter nbconvert --to notebook --execute ssl_pretrain.ipynb \
  --output ssl_pretrain_executed.ipynb \
  --output-dir /users/jmatthia/deep_learning/code \
  --debug

# Supervised training and transfer learning on the Intel Images dataset

```python
# complete loop with different seeds
seeds = [0, 1, 2]
batch_size_parameter = 64
epochs_parameter = 60
max_increase_loss = 3
val_size = 2000

loss_function = nn.CrossEntropyLoss()
learning_rate_parameter = 0.001
momentum_parameter = 0.9
weight_decay_parameter = 1e-4
device_parameter = torch.device("cuda" if torch.cuda.is_available() else "cpu")``` 
```

# Cluster submission Intel datasets, with our 5 different models

#!/bin/bash
#SBATCH --dependency=afterok:28149956 
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-171
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --job-name=ssl_pretrain
#SBATCH --output=/users/jmatthia/deep_learning/code/ssl_intel_loader_%j.out
#SBATCH --error=/users/jmatthia/deep_learning/code/ssl_intel_loader_%j.err

# shell + conda
source ~/.bashrc
conda activate /users/jmatthia/deep_learning/env/dl_env_2

# go to code dir (important for relative paths)
cd /users/jmatthia/deep_learning/code

# execute notebook non-interactively
jupyter nbconvert \
  --to notebook \
  --execute intel_loader.ipynb \
  --output intel_loader_executed.ipynb \
  --output-dir /users/jmatthia/deep_learning/code \
  --debug


