#!/bin/bash
#SBATCH --job-name=create_env_3dinfomax
#SBATCH --output=/home/jmatthi2/deep_learning/COMET/3dinfomax/logs/create_env_%j.out
#SBATCH --error=/home/jmatthi2/deep_learning/COMET/3dinfomax/logs/create_env_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# create_env_3dinfomax.sh
# Clones 3D Infomax repo and builds conda env manually.
# Uses Python 3.9 + PyTorch 1.13.1 for compatibility with the repo's codebase.
#
# Usage:
#   mkdir -p /home/jmatthi2/deep_learning/COMET/3dinfomax/logs
#   sbatch create_env_3dinfomax.sh

WORK_DIR="/home/jmatthi2/deep_learning/COMET/3dinfomax"
REPO_DIR="${WORK_DIR}/3DInfomax"
ENV_PATH="/home/jmatthi2/deep_learning/env_3dinfomax"

module load anaconda3/2024.02-1

echo "============================================"
echo "Setting up 3D Infomax"
echo "============================================"

# Clone repo if not already present
if [ ! -d "${REPO_DIR}" ]; then
    echo "Cloning 3DInfomax repo..."
    cd "${WORK_DIR}"
    git clone https://github.com/HannesStark/3DInfomax.git
else
    echo "Repo already cloned at ${REPO_DIR}"
fi

# Remove old broken env if it exists
if [ -d "${ENV_PATH}" ]; then
    echo "Removing old environment..."
    conda env remove --prefix "${ENV_PATH}" -y 2>/dev/null
    rm -rf "${ENV_PATH}"
fi

echo ""
echo "Creating conda environment at ${ENV_PATH}..."
echo ""

# Step 1: Python 3.9 (collections.MutableMapping still works, broad compat)
conda create --prefix "${ENV_PATH}" python=3.9 -y

# Step 2: Activate
conda activate "${ENV_PATH}"

# Step 3: PyTorch 1.13.1 CPU (matches the era the repo was written in)
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Step 4: PyG extensions matched to torch 1.13.1 CPU
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
pip install torch_geometric==2.3.1

# Step 5: DGL CPU (1.1.x doesn't need torchdata)
pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/repo.html

# Step 6: OGB pinned to version that has DglPCQM4MDataset
pip install ogb==1.3.5

# Step 7: RDKit
conda install -c conda-forge rdkit -y

# Step 8: Other deps
pip install icecream seaborn pyyaml matplotlib numpy pandas scikit-learn tensorboard

# Step 9: Patch the repo's ogb import (1.3.5 may still miss it)
cd "${REPO_DIR}"
python -c "
for fname in ['inference.py', 'train.py']:
    with open(fname, 'r') as f:
        text = f.read()
    text = text.replace(
        'from ogb.lsc import DglPCQM4MDataset, PCQM4MEvaluator',
        'try:\n    from ogb.lsc import DglPCQM4MDataset, PCQM4MEvaluator\nexcept ImportError:\n    DglPCQM4MDataset = None\n    PCQM4MEvaluator = None'
    )
    with open(fname, 'w') as f:
        f.write(text)
print('Patched ogb imports.')
"

echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_scatter; print('torch_scatter: OK')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import dgl; print(f'DGL: {dgl.__version__}')"
python -c "from rdkit import Chem; print('RDKit: OK')"
python -c "import ogb; print(f'OGB: {ogb.__version__}')"
python -c "import icecream; print('icecream: OK')"
python -c "import yaml; print('PyYAML: OK')"

echo ""
echo "============================================"
echo "Environment created at: ${ENV_PATH}"
echo ""
echo "Activate with:"
echo "  module load anaconda3/2024.02-1"
echo "  conda activate ${ENV_PATH}"
echo "============================================"
