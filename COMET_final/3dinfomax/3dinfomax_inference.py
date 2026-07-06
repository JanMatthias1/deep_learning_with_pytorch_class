"""
3D Infomax lipid embedding extraction.

Uses the repo's built-in inference pipeline:
  1. Reads SMILES from a CSV (component_type,name,smiles)
  2. Writes them to dataset/inference_smiles.txt (the format inference.py expects)
  3. Calls inference.py which loads the pretrained PNA checkpoint,
     runs forward pass, and saves fingerprints.pt
  4. Converts fingerprints.pt -> .npy dict {smiles: embedding} to match SchNet format

Usage:
  python 3dinfomax_inference.py \
      --csv_path /path/to/lance_lipid_smiles.csv \
      --repo_dir /path/to/3DInfomax \
      --output_path /path/to/3dinfomax_lipid_embeddings.npy
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D Infomax embeddings for lipid SMILES"
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV with columns: component_type,name,smiles")
    parser.add_argument("--repo_dir", type=str, required=True,
                        help="Path to cloned 3DInfomax repo")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output .npy file for embeddings dict")
    parser.add_argument("--config", type=str,
                        default="configs_clean/fingerprint_inference.yml",
                        help="Config file for inference (default: fingerprint_inference.yml)")
    args = parser.parse_args()

    # ---- Step 1: Read SMILES from CSV ----
    import csv
    smiles_list = []
    with open(args.csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row["smiles"].strip()
            if smi:
                smiles_list.append(smi)

    print(f"Read {len(smiles_list)} SMILES from {args.csv_path}")

    # ---- Step 2: Write SMILES to inference_smiles.txt ----
    smiles_txt_path = os.path.join(args.repo_dir, "dataset", "inference_smiles.txt")
    os.makedirs(os.path.dirname(smiles_txt_path), exist_ok=True)

    with open(smiles_txt_path, "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")

    print(f"Wrote SMILES to {smiles_txt_path}")

    # ---- Step 3: Run the repo's inference.py ----
    print("\nRunning 3D Infomax inference...")
    cmd = [
        sys.executable, "inference.py",
        f"--config={args.config}",
        f"--smiles_txt_path={smiles_txt_path}",
    ]
    result = subprocess.run(cmd, cwd=args.repo_dir, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(1)

    # ---- Step 4: Convert fingerprints.pt to .npy dict ----
    fp_path = os.path.join(args.repo_dir, "dataset", "fingerprints.pt")
    if not os.path.exists(fp_path):
        print(f"ERROR: fingerprints.pt not found at {fp_path}")
        sys.exit(1)

    data = torch.load(fp_path, map_location="cpu")
    fingerprints = data["fingerprints"]  # shape: [n_molecules, embed_dim]

    print(f"\nFingerprints shape: {fingerprints.shape}")
    print(f"Embedding dim: {fingerprints.shape[1]}")

    if fingerprints.shape[0] != len(smiles_list):
        print(f"WARNING: Got {fingerprints.shape[0]} embeddings "
              f"for {len(smiles_list)} SMILES")

    # Build dict: smiles -> numpy array
    embeddings = {}
    for i, smi in enumerate(smiles_list):
        if i < fingerprints.shape[0]:
            embeddings[smi] = fingerprints[i].detach().numpy()

    np.save(args.output_path, embeddings)
    print(f"\nSaved {len(embeddings)} embeddings to {args.output_path}")
    print(f"Embedding dim: {fingerprints.shape[1]}")

    # Print summary
    for i, (smi, emb) in enumerate(embeddings.items()):
        print(f"  [{i+1}] dim={emb.shape[0]}  {smi[:50]}...")


if __name__ == "__main__":
    main()
