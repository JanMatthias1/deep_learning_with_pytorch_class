# Lipid Nanoparticle Design with Composite Material Transformer (COMET) & GNN Augmentation
===================================================================
This repository contains the code for **COMET**, a transformer-based model for predicting Lipid Nanoparticle (LNP) transfection efficacy, as well as extended architectures that fuse Graph Neural Network (GNN) embeddings (GIN, SchNet, 3D Infomax) into the frozen Uni-Mol backbone via element-wise vector addition.

## Setup and Installation

### Requirements
- Python 3.10
- NVIDIA GPU with modern drivers (CUDA 11.6 or newer; CUDA 12.1+ recommended)
- Anaconda or Miniconda

### Creating the Environment
1. Load the required modules and create a new Anaconda environment:
    ```bash
    conda create -n comet_env python=3.10 -y
    conda activate comet_env
    ```

2. Install PyTorch customized for your machine's CUDA version. (Example for CUDA 12.1):
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
    *(For legacy CUDA 11.6: `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia`)*

3. Install Dependencies (CRITICAL: NumPy must be pinned to 1.26.4 to avoid silent crashes):
    ```bash
    pip install numpy==1.26.4 pandas==2.2.2 numexpr==2.10.1 scipy==1.13.0
    pip install scikit-learn==1.4.2 tensorboard==2.16.2 tqdm==4.64.1
    pip install lmdb==1.4.1 rdkit==2023.9.5 biopython==1.83 networkx==3.3
    ```

4. Install Uni-Core (Required for Uni-Mol):
    ```bash
    pip install [https://github.com/dptech-corp/Uni-Core/releases/download/0.0.2/unicore-0.0.1+cu116torch1.13.1-cp310-cp310-linux_x86_64.whl](https://github.com/dptech-corp/Uni-Core/releases/download/0.0.2/unicore-0.0.1+cu116torch1.13.1-cp310-cp310-linux_x86_64.whl)
    ```

5. Set the PYTHONPATH (Because `unimol` is treated as a local module, you must tell Python where it is located before running any scripts):
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    ```
    *(Tip: Add this line to your `~/.bashrc` file to avoid running it every time).*

## Data Preprocessing
Preprocessing is done to make lmdb datasets (stored in `experiments/processed_data_dirs/`) from json files (stored in `experiments/data_json/`). Scripts for generating processed datasets:
- `experiments/preprocess_data_LANCE.ipynb`: Processes data for LNPs' efficacy on DC2.4 and B16-F10 cells.
- `experiments/preprocess_data_CACO2.ipynb`: Includes CACO2 cell transfection data.
- `experiments/preprocess_data_stability.ipynb`: Processes data for lyophilized LNPs.

## GNN Augmentations & Precomputed Embeddings
This project fuses GNN features into the Uni-Mol pipeline. The GNN embeddings are precomputed offline to save compute overhead during training and are located in the root directory:
- GIN: `GIN/lance_lipid_gin_embeddings.npy`
- SchNet: `schnet/schnet_lipid_embeddings.npy`
- 3D Infomax: `3dinfomax/3dinfomax_lipid_embeddings.npy`

## Training
To run the comparative analysis between the baseline and the GNN-augmented variants, navigate to the `experiments/` directory and execute the respective grid search scripts in the background. These scripts automatically handle distributed data parallel (DDP) scaling, freeze the Uni-Mol backbone, and execute early stopping (patience=20).

```bash
cd experiments/

# Run Baseline COMET
python run_baseline.py > logs/baseline_output.log 2>&1 &

# Run GIN Augmented COMET
python run_gin.py > logs/gin_output.log 2>&1 &

# Run SchNet Augmented COMET
python run_schnet.py > logs/schnet_output.log 2>&1 &

# Run 3D Infomax Augmented COMET
python run_infomax.py > logs/infomax_output.log 2>&1 &

(Check the logs live using tail -f logs/[log_name].log)
```

## Inference
The grid search scripts above automatically execute inference on the held-out test splits after training concludes. Predictions are stored as `.pkl` files ending in `.out.pkl`.

Inference results are outputted at location: `experiments/infer_results/infer_[Model_Name]_Run/`

*(Legacy inference scripts for PBAE and general LNP deployment are also available as `experiments/inference_script_LANCE_*.py`)*

## Troubleshooting

- **"Port 10086 is already in use" / DDP Hanging:** If a script is killed (`Ctrl+C`), PyTorch DDP often leaves "zombie" processes running on the GPU, which block the port for future runs. Run this command to clear them:
    ```bash
    pkill -9 -u [your_username] -f python
    ```

- **LMDB Not Found Error:** Ensure `--concat-datasets` is present in your subprocess call. This allows the model to search subdirectories for `train.lmdb` and `valid.lmdb`.

- **Gradient Overflow:** Ensure you are using the optimized Small architecture (e.g., `--lnp-encoder-layers 8 --lnp-encoder-embed-dim 256`) and that `--fp16` is enabled to speed up batch times and stabilize training.

## Key Files & Folders
- `experiments/run_*.py`: Automated grid search and training scripts for the specific model architectures.
- `experiments/task_schemas/`: Stores task schema files: dictionary file to specify key information of multiple datasets. The keys are names of the datasets, value is a dictionary containing `tasks_schema_path`, `component_types_schema_path` and `np_prop_schema_path` as keys.
- `experiments/data_json/`: Stores raw LNP data.
- `experiments/processed_data_dirs/`: Stores processed lmdb datasets for training and inference.
- `experiments/save_demo/`: Output directory for trained model weights (`checkpoint_best.pt`).
- `unimol/`: Core logic for model operations (includes customized `train_np.py` and `infer_np.py` for element-wise fusion).
- `ckp/`: Pretrained Uni-Mol molecular backbone weights (`mol_pre_no_h_220816.pt`).

## Pretrained Weights
Available pretrained weights for different models are listed under `experiments/weights/`. These can be used directly for deploying models and running inference scripts.

### Note
Code has been tested on Linux and installation is expected to take less than 1 hour on a typical computer with an internet connection. Standard training runs take roughly 5-6 hours per configuration on an RTX 4070.
