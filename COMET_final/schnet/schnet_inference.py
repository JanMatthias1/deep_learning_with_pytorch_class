"""
SchNet embedding extraction for COMET lipid components.

Analogous to gin_inference.py but uses a pretrained SchNet model (QM9)
instead of a pretrained GIN (ChEMBL/supervised_contextpred).

Key differences from GIN pipeline:
  - SchNet requires 3D atomic coordinates (not just 2D graph topology)
  - 3D conformers generated via RDKit ETKDG + MMFF optimization
  - Pretrained weights from PyG's SchNet.from_qm9_pretrained()
  - Output embedding is 128-dim (vs GIN's 300-dim)
  - Dot-disconnected SMILES (e.g. "CCCC.N") are split into fragments,
    each fragment is embedded separately, then averaged weighted by
    atom count.

Usage:
  python schnet_inference.py --csv_path <smiles.csv> --output_path schnet_embeddings.npy
  # python schnet_inference.py --json_path <comet_data.json> --output_path schnet_embeddings.npy
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import SchNet
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import QM9


def smiles_to_3d(smiles, max_attempts=5000):
    """
    Convert a SMILES string to (atomic_numbers, positions).
    Returns None if conformer generation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.maxAttempts = max_attempts
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        params.useRandomCoords = True
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            return None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    conf = mol.GetConformer()
    positions = conf.GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    return np.array(atomic_numbers), positions


def extract_representation(model, z, pos, batch):
    """
    Run SchNet forward pass up to (but not including) the output head.

    SchNet forward:
      h = embedding(z)
      for interaction in interactions:
          h = h + interaction(h, edge_index, edge_weight, edge_attr)
      h = lin1(h) -> act -> lin2(h)    <-- output head, SKIPPED
      readout                           <-- we do our own mean pool

    Returns mean-pooled 128-dim graph embedding.
    """
    h = model.embedding(z)
    edge_index, edge_weight = model.interaction_graph(pos, batch)
    edge_attr = model.distance_expansion(edge_weight)

    for interaction in model.interactions:
        h = h + interaction(h, edge_index, edge_weight, edge_attr)

    graph_repr = global_mean_pool(h, batch)
    return graph_repr.squeeze(0).numpy()


def embed_smiles(smiles, model, max_attempts=5000):
    """
    Embed a single SMILES string.

    Handles dot-disconnected fragments (e.g. "CCCC.N") by embedding
    each fragment separately, then combining via weighted average
    by atom count.

    Returns (embedding, n_atoms) or (None, 0) on failure.
    """
    fragments = smiles.split('.')

    if len(fragments) == 1:
        result = smiles_to_3d(smiles, max_attempts=max_attempts)
        if result is None:
            return None, 0
        atomic_numbers, positions = result
        z = torch.tensor(atomic_numbers, dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float32)
        batch = torch.zeros(len(atomic_numbers), dtype=torch.long)
        with torch.no_grad():
            emb = extract_representation(model, z, pos, batch)
        return emb, len(atomic_numbers)

    # Multi-fragment: embed each, weighted average by atom count
    fragment_embs = []
    fragment_weights = []

    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue
        result = smiles_to_3d(frag, max_attempts=max_attempts)
        if result is None:
            continue
        atomic_numbers, positions = result
        z = torch.tensor(atomic_numbers, dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float32)
        batch = torch.zeros(len(atomic_numbers), dtype=torch.long)
        with torch.no_grad():
            emb = extract_representation(model, z, pos, batch)
        fragment_embs.append(emb)
        fragment_weights.append(len(atomic_numbers))

    if len(fragment_embs) == 0:
        return None, 0

    weights = np.array(fragment_weights, dtype=np.float64)
    weights = weights / weights.sum()
    combined = sum(w * e for w, e in zip(weights, fragment_embs))
    total_atoms = sum(fragment_weights)

    return combined, total_atoms


# --- Future: extract SMILES directly from COMET JSON ---
# def collect_unique_smiles_from_json(json_path):
#     """Extract all unique SMILES from a COMET JSON file."""
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     smiles_set = set()
#     for key, entry in data.items():
#         for comp in entry['components']:
#             smiles_set.add(comp['smi'])
#     return sorted(smiles_set)


def collect_unique_smiles_from_csv(csv_path):
    """Extract SMILES from a CSV (assumes 'smiles' column)."""
    df = pd.read_csv(csv_path)
    return df['smiles'].tolist()


def load_pretrained_schnet(qm9_data_dir, target=7):
    """
    Load pretrained SchNet from PyG's QM9 pretrained weights.

    target=7 = U0 (internal energy at 0K). The representation weights
    are independent of the target head -- we discard the output head.

    Requires: pip install schnetpack (PyG uses it internally to load weights)
    """
    print(f"Loading QM9 dataset from {qm9_data_dir}...")
    dataset = QM9(root=qm9_data_dir)
    print("Loading pretrained SchNet weights...")
    model, _ = SchNet.from_qm9_pretrained(qm9_data_dir, dataset, target)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Extract SchNet embeddings for COMET lipid molecules'
    )

    # --- Active: CSV mode ---
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV with smiles column')

    # --- Future: JSON mode ---
    # parser.add_argument('--json_path', type=str, required=True,
    #                     help='Path to COMET JSON data file')

    parser.add_argument('--output_path', type=str, required=True,
                        help='Output .npy path for embeddings dict')
    parser.add_argument('--qm9_data_dir', type=str, default='./qm9_data',
                        help='Directory to download/cache QM9 dataset')
    parser.add_argument('--target', type=int, default=7,
                        help='QM9 target for pretrained model (default: 7 = U0)')
    parser.add_argument('--max_attempts', type=int, default=5000,
                        help='Max attempts for ETKDG conformer generation')
    args = parser.parse_args()

    # --- Active: CSV mode ---
    smiles_list = collect_unique_smiles_from_csv(args.csv_path)
    print(f"Found {len(smiles_list)} SMILES from CSV")

    # --- Future: JSON mode ---
    # smiles_list = collect_unique_smiles_from_json(args.json_path)
    # print(f"Found {len(smiles_list)} unique SMILES from JSON")

    # Load pretrained SchNet
    model = load_pretrained_schnet(args.qm9_data_dir, target=args.target)
    print("Model loaded.\n")

    embeddings = {}
    failed = []

    for i, smiles in enumerate(smiles_list):
        has_fragments = '.' in smiles
        emb, n_atoms = embed_smiles(smiles, model,
                                     max_attempts=args.max_attempts)

        if emb is None:
            print(f"[{i+1}/{len(smiles_list)}] FAILED  {smiles[:60]}...")
            failed.append(smiles)
            continue

        embeddings[smiles] = emb

        frag_note = ""
        if has_fragments:
            n_frags = len([f for f in smiles.split('.') if f.strip()])
            frag_note = f"  ({n_frags} fragments, averaged)"

        print(f"[{i+1}/{len(smiles_list)}] OK  dim={emb.shape[0]}  "
              f"atoms={n_atoms}  {smiles[:50]}...{frag_note}")

    np.save(args.output_path, embeddings)
    print(f"\nDone. {len(embeddings)} embeddings saved to {args.output_path}")
    print(f"{len(failed)} failed.")
    if failed:
        print("Failed SMILES:")
        for s in failed:
            print(f"  {s}")


if __name__ == "__main__":
    main()
