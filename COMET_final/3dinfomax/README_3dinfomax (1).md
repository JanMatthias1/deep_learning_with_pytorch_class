# 3D Infomax Lipid Embedding Pipeline

Extract molecular embeddings from lipid SMILES using a pretrained 3D Infomax (PNA) model, then run pairwise cosine similarity analysis.

---

## Prerequisites

- SLURM cluster with `anaconda3/2024.02-1` module available
- Internet access from compute nodes (for `git clone` and pip installs)
- Your lipid SMILES in a CSV at:
  ```
  /home/jmatthi2/deep_learning/COMET/3dinfomax/lance_lipid_smiles.csv
  ```
  Required columns: `component_type`, `name`, `smiles`

---

## Directory Structure

After setup, the working directory looks like:

```
/home/jmatthi2/deep_learning/COMET/3dinfomax/
├── logs/                          # SLURM stdout/stderr
├── 3DInfomax/                     # Cloned repo (created by step 1)
├── lance_lipid_smiles.csv         # Your input (you provide this)
├── 3dinfomax_lipid_embeddings.npy # Output embeddings (created by step 2)
├── 3dinfomax_similarity_matrix.png# Output plot (created by step 3)
├── create_env_3dinfomax.sh
├── run_3dinfomax_inference.sh
├── run_simcheck_3dinfomax.sh
├── 3dinfomax_inference.py
└── similarity_check_3dinfomax.py
```

---

## Step 0 — Stage scripts and create log directory

Copy all scripts into the working directory, then:

```bash
mkdir -p /home/jmatthi2/deep_learning/COMET/3dinfomax/logs
```

All scripts must be in `/home/jmatthi2/deep_learning/COMET/3dinfomax/`.

---

## Step 1 — Build the conda environment

Clones the 3DInfomax repo and installs all dependencies (PyTorch 1.13.1 CPU, PyG, DGL, OGB, RDKit).

```bash
sbatch create_env_3dinfomax.sh
```

**Runtime:** ~20–40 min depending on node and network speed.

Check completion:
```bash
cat /home/jmatthi2/deep_learning/COMET/3dinfomax/logs/create_env_*.out | tail -20
```

Expected tail output:
```
PyTorch: 1.13.1+cpu
torch_scatter: OK
PyG: 2.3.1
DGL: 1.1.3
RDKit: OK
OGB: 1.3.5
icecream: OK
PyYAML: OK
Environment created at: /home/jmatthi2/deep_learning/env_3dinfomax
```

---

## Step 2 — Extract embeddings

Reads `lance_lipid_smiles.csv`, runs the pretrained PNA model, saves embeddings as a `{smiles: np.array}` dict.

```bash
sbatch run_3dinfomax_inference.sh
```

**Runtime:** ~5–20 min (CPU only).

Check completion:
```bash
cat /home/jmatthi2/deep_learning/COMET/3dinfomax/logs/inference_*.out | tail -10
```

Expected tail output:
```
Saved N embeddings to .../3dinfomax_lipid_embeddings.npy
Embedding dim: 70
  [1] dim=70  <smiles>...
  ...
Done.
```

Output file: `3dinfomax_lipid_embeddings.npy`

---

## Step 3 — Similarity check

Computes pairwise cosine similarity across all lipid embeddings, prints the matrix and off-diagonal stats, warns on near-identical pairs, and saves a heatmap plot.

```bash
sbatch run_simcheck_3dinfomax.sh
```

**Runtime:** <5 min.

Check completion:
```bash
cat /home/jmatthi2/deep_learning/COMET/3dinfomax/logs/simcheck_*.out
```

Expected output includes:
```
=== Cosine Similarity Matrix ===
...
=== Off-diagonal stats ===
  Mean:   X.XXXX
  Std:    X.XXXX
  Min:    X.XXXX
  Max:    X.XXXX
  Median: X.XXXX

=== Similarity Warnings (sim > 0.99) ===
None found.

Plot saved to .../3dinfomax_similarity_matrix.png
```

---

## Outputs

| File | Description |
|---|---|
| `3dinfomax_lipid_embeddings.npy` | Dict `{smiles: np.array}`, shape `[n, 70]` |
| `3dinfomax_similarity_matrix.png` | Pairwise cosine similarity heatmap |

---

## Loading embeddings downstream

```python
import numpy as np

embeddings = np.load("3dinfomax_lipid_embeddings.npy", allow_pickle=True).item()
# embeddings: dict {smiles_str: np.array of shape (70,)}

smiles = list(embeddings.keys())
matrix = np.stack(list(embeddings.values()))  # shape: [n_lipids, 70]
```


