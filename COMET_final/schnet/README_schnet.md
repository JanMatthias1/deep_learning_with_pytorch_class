# SchNet Embedding Extraction for COMET

## Overview

This pipeline extracts molecular embeddings from a **pretrained SchNet model** for use as auxiliary GNN features in the COMET LNP property prediction framework. It is the 3D-geometry-aware counterpart to the existing GIN embedding pipeline.

Each unique lipid component SMILES in the dataset gets a **128-dimensional embedding vector**. These embeddings are saved as a dictionary (`{smiles_string: 128d_numpy_array}`) in a `.npy` file, which COMET later uses to look up per-component representations during training.

---

## Background

### What is SchNet?

SchNet is a continuous-filter convolutional neural network for modeling quantum interactions between atoms. Unlike 2D graph neural networks (GIN, GCN, etc.) that operate on molecular topology (atoms + bonds), SchNet operates on **3D atomic coordinates** and uses **interatomic distances** as features. This allows it to capture spatial/geometric information that 2D methods cannot.

**Paper:** Schütt et al., "SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions" (NeurIPS 2017)

### Comparison with GIN pipeline

| | GIN (existing) | SchNet (this pipeline) |
|---|---|---|
| **Weights** | `supervised_contextpred.pth` (manual download) | Auto-downloaded by PyG |
| **Trained by** | Hu et al. (Stanford) | Schütt et al. (TU Berlin) |
| **Pretraining data** | ~2M molecules from ChEMBL | ~134K molecules from QM9 |
| **Pretraining task** | Biological activity prediction | Quantum energy (U0) prediction |
| **Input** | 2D molecular graph (atom types + bond types) | 3D coordinates + atomic numbers |
| **Features used** | Bond topology, chirality | Interatomic distances |
| **Output dim** | 300 | 128 |
| **Domain gap to lipids** | Moderate (drug-like molecules) | Larger (QM9 molecules have ≤9 heavy atoms) |

### What we extract

SchNet's forward pass has two stages:

```
STAGE 1 — REPRESENTATION (we keep this)

  Atomic numbers + 3D positions
           │
           ▼
  Embedding layer: atomic number → 128-dim vector per atom
           │
           ▼
  Compute pairwise interatomic distances (within cutoff)
           │
           ▼
  Interaction block 1 ──► atoms exchange information based on 3D distances
  Interaction block 2
  Interaction block 3
  Interaction block 4
  Interaction block 5
  Interaction block 6
           │
           ▼
  128-dim vector per atom (learned 3D chemical environment encoding)
           │
           ▼
  Mean pool over all atoms → single 128-dim vector per molecule  ← WE EXTRACT THIS
```

```
STAGE 2 — OUTPUT HEAD (we discard this)

  128-dim per atom → Linear → Activation → Linear → scalar per atom
           │
           ▼
  Sum over atoms → single scalar (predicted U0 energy)
```

We cut off after Stage 1. The 128-dim per-atom vectors encode the 3D chemical
environment as learned from quantum chemistry data. The output head (Stage 2)
just collapses these to a single energy scalar, which is task-specific and
useless for our purposes.

---

## Pipeline

### Step 1: SMILES → 3D Conformer

Since SchNet requires 3D coordinates, we generate conformers from SMILES using RDKit:
1. Parse SMILES with `Chem.MolFromSmiles()`
2. Add explicit hydrogens with `Chem.AddHs()`
3. Generate 3D coordinates with `AllChem.EmbedMolecule()` using ETKDG v3
4. Optimize geometry with MMFF94 force field

**Dot-disconnected SMILES** (e.g. `CCCC...OC(=O)CCCCC.N`) contain multiple
disconnected fragments. These are split and embedded separately, then combined
via atom-count-weighted averaging. This avoids artifacts from RDKit placing
disconnected fragments at arbitrary distances in 3D space.

### Step 2: 3D Conformer → SchNet → 128-dim Embedding

The pretrained SchNet model processes atomic numbers and 3D positions through
6 interaction blocks, producing 128-dim per-atom features. These are
mean-pooled over atoms to produce one 128-dim graph-level embedding.

### Step 3: Save Embeddings

Output is a `.npy` file containing a Python dictionary:
```python
{
    "OCCCCN(CCCCCCOC(...)=O)CCCCCCOC(...)=O": np.array([...]),  # 128-dim
    "CCCCCCCCC=CCCCCCCCC(=O)OCC(...)OC(=O)...": np.array([...]),  # 128-dim
    ...
}
```

One entry per unique SMILES. COMET looks up embeddings by SMILES key and
combines them per-formulation using molar ratios in the LNP encoder.

---

## File Structure

```
schnet_encoding/
├── schnet_inference.py          # Main embedding extraction script
├── similarity_check_schnet.py   # Cosine similarity validation
├── create_env_schnet.sh         # Conda environment setup
├── run_schnet_inference.sh      # SLURM job: extract embeddings
├── run_similarity_check.sh      # SLURM job: similarity matrix
├── lance_lipid_smiles.csv       # Input: CSV with 'smiles' column
├── schnet_lipid_embeddings.npy  # Output: embeddings dictionary
├── schnet_similarity_matrix.png # Output: similarity heatmap
└── qm9_data/                    # Auto-created: cached QM9 dataset + weights
```

---

## Setup and Usage

### 1. Create the environment

```bash
bash create_env_schnet.sh
```

This creates a conda environment at `/home/jmatthi2/deep_learning/env_schnet` with:
- Python 3.10
- PyTorch (CPU)
- PyTorch Geometric (`torch_geometric`) — contains the SchNet model class
- SchNetPack — required internally by PyG to load pretrained weights
- RDKit — for SMILES parsing and 3D conformer generation
- pandas, numpy, matplotlib

### 2. Prepare input data

Place `lance_lipid_smiles.csv` in the working directory. The CSV must have a
`smiles` column. Optionally include a `name` column for labeled similarity plots.

### 3. Extract embeddings

```bash
# Edit paths in run_schnet_inference.sh, then:
sbatch run_schnet_inference.sh
```

Or run directly:
```bash
conda activate /home/jmatthi2/deep_learning/env_schnet

python schnet_inference.py \
    --csv_path lance_lipid_smiles.csv \
    --output_path schnet_lipid_embeddings.npy \
    --qm9_data_dir ./qm9_data \
    --target 7 \
    --max_attempts 5000
```

**First run** downloads QM9 (~700MB) and pretrained weights. These are cached
in `qm9_data/` for subsequent runs.

**Expected output:**
```
Found 21 SMILES from CSV
Loading QM9 dataset from ./qm9_data...
Loading pretrained SchNet weights...
Model loaded.

[1/21] OK  dim=128  atoms=156  OCCCCN(CCCCCCOC(C(CCCCCC)CCCCCCCC)=O)CC...
[2/21] OK  dim=128  atoms=203  CCCCC/C=C\C/C=C\CCCCCCCCC1(OC(CCN(C)C)...
...
[21/21] OK  dim=128  atoms=178  CCCCCCCCCCCCCC(=O)OCC(...)CCCCCCCCCCCCC.N  (2 fragments, averaged)

Done. 21 embeddings saved to schnet_lipid_embeddings.npy
0 failed.
```

### 4. Validate embeddings

```bash
sbatch run_similarity_check.sh
```

Or directly:
```bash
python similarity_check_schnet.py \
    --embeddings_path schnet_lipid_embeddings.npy \
    --names_csv lance_lipid_smiles.csv \
    --output_path schnet_similarity_matrix.png
```

This produces:
- Console warnings for any near-identical embeddings (cosine sim > 0.99)
- A pairwise cosine similarity heatmap saved as a PNG

---

## CLI Arguments

### schnet_inference.py

| Argument | Required | Default | Description |
|---|---|---|---|
| `--csv_path` | Yes | — | Path to CSV with `smiles` column |
| `--output_path` | Yes | — | Output `.npy` file path |
| `--qm9_data_dir` | No | `./qm9_data` | Where to cache QM9 data + weights |
| `--target` | No | `7` | QM9 property target (7 = U0). Representation weights are the same regardless of target — only the discarded output head differs |
| `--max_attempts` | No | `5000` | Max ETKDG conformer generation attempts. Increase if large lipids fail |

### similarity_check_schnet.py

| Argument | Required | Default | Description |
|---|---|---|---|
| `--embeddings_path` | Yes | — | Path to `.npy` embeddings dict |
| `--names_csv` | No | None | CSV with `smiles`,`name` columns for plot labels |
| `--output_path` | No | auto | Output PNG path (defaults to `<embeddings>_similarity_matrix.png`) |
| `--sim_floor` | No | `0.8` | Lower bound of colormap |
| `--warn_threshold` | No | `0.99` | Cosine similarity threshold for warnings |

---

## Integration with COMET

When integrating these embeddings into COMET, set `--gnn-embed-dim 128` (instead
of 300 for GIN). The `lnp_rep_proj` projection layer handles mapping from
`lnp_component_rep_dim + gnn_embed_dim` to COMET's internal dimension, so the
GNN embedding dimension is flexible.

The JSON mode for reading SMILES directly from COMET data files is implemented
but commented out in `schnet_inference.py`. Uncomment to use.

---

## Known Limitations

1. **Domain shift**: QM9 contains small organic molecules with ≤9 heavy atoms
   (C, N, O, F only). Lipid molecules are much larger (50+ heavy atoms) with
   different chemical space. Transfer quality is uncertain.

2. **Conformer sensitivity**: SchNet embeddings depend on the specific 3D
   conformer generated. Different conformers of the same molecule will produce
   slightly different embeddings. We fix `randomSeed=42` for reproducibility.

3. **Large molecule failures**: ETKDG may fail to generate conformers for very
   large or highly flexible lipids. The script falls back to random coordinate
   initialization and increases `maxAttempts`, but some molecules may still fail.

4. **Hydrogens included**: RDKit adds explicit hydrogens before conformer
   generation (required for realistic 3D geometry). This increases atom count
   substantially. SchNet processes all atoms including H.

---

## Pretrained Model Details

- **Architecture**: SchNet with 128 hidden channels, 128 filters, 6 interaction
  blocks, 50 Gaussian RBFs, 10.0 Å cutoff, cosine cutoff function
- **Training data**: QM9 (130,831 small organic molecules)
- **Training target**: U0 (internal energy at 0K)
- **Source**: Weights hosted by SchNetPack authors, downloaded and converted
  by PyTorch Geometric's `SchNet.from_qm9_pretrained()` method
- **No manual download required**: weights are fetched and cached automatically
