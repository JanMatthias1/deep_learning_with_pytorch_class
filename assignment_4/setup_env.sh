#!/bin/bash
# setup_env.sh
# Creates and configures the Python virtual environment for the UCF50 assignment.
# Run this once from the project root before training or testing.
#
# Usage:
#   bash setup_env.sh

set -e

PYTHON=${PYTHON:-python3}
ENV_DIR="ucf50_env"

# ── 1. Check Python version ────────────────────────────────────────────────────
PY_VERSION=$($PYTHON -c "import sys; print(sys.version_info[:2])")
echo "[setup] Python version: $PY_VERSION"
$PYTHON -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'" \
    || { echo "ERROR: Python 3.10 or higher is required."; exit 1; }

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ -d "$ENV_DIR" ]; then
    echo "[setup] Virtual environment '$ENV_DIR' already exists — skipping creation."
else
    echo "[setup] Creating virtual environment: $ENV_DIR"
    $PYTHON -m venv "$ENV_DIR"
fi

# ── 3. Activate and install dependencies ──────────────────────────────────────
echo "[setup] Installing dependencies from requirements.txt ..."
"$ENV_DIR/bin/pip" install --upgrade pip --quiet
"$ENV_DIR/bin/pip" install -r requirements.txt --quiet

# ── 4. Verify critical imports ─────────────────────────────────────────────────
echo "[setup] Verifying installation ..."
"$ENV_DIR/bin/python" - <<'EOF'
import torch
import torchvision
import cv2
import sklearn
import numpy
import PIL
import tqdm
import matplotlib
import seaborn

print(f"  torch        {torch.__version__}")
print(f"  torchvision  {torchvision.__version__}")
print(f"  opencv       {cv2.__version__}")
print(f"  scikit-learn {sklearn.__version__}")
print(f"  numpy        {numpy.__version__}")
print(f"  Pillow       {PIL.__version__}")

cuda = torch.cuda.is_available()
print(f"  CUDA available: {cuda}")
if cuda:
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "[setup] Done. Activate the environment with:"
echo "        source $ENV_DIR/bin/activate"
echo ""
echo "Then run training with:  bash train.sh"
echo "Then run testing with:   bash test.sh"
