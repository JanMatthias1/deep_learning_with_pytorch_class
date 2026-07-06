"""
Pairwise cosine similarity check for 3D Infomax lipid embeddings.

Same analysis as similarity_check_schnet.py, adapted for 3D Infomax
embedding dimensionality (PNA default: 70-dim, but depends on config).

Usage:
  python similarity_check_3dinfomax.py --embeddings_path 3dinfomax_lipid_embeddings.npy
  python similarity_check_3dinfomax.py --embeddings_path 3dinfomax_lipid_embeddings.npy --names_csv lance_lipid_smiles.csv
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import norm


def main():
    parser = argparse.ArgumentParser(
        description="Cosine similarity matrix for 3D Infomax embeddings"
    )
    parser.add_argument("--embeddings_path", type=str, required=True,
                        help="Path to .npy embeddings dict")
    parser.add_argument("--names_csv", type=str, default=None,
                        help="Optional CSV with smiles,name columns for labels")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for similarity matrix plot")
    parser.add_argument("--sim_floor", type=float, default=-1.0,
                        help="Lower bound for colormap (default: 0.8)")
    parser.add_argument("--warn_threshold", type=float, default=0.99,
                        help="Threshold for near-identical warning (default: 0.99)")
    args = parser.parse_args()

    # Load embeddings
    embeddings = np.load(args.embeddings_path, allow_pickle=True).item()
    smiles_list = list(embeddings.keys())
    emb_matrix = np.stack(list(embeddings.values()))
    n = len(smiles_list)

    print(f"Loaded {n} embeddings, dim={emb_matrix.shape[1]}")

    # Build labels
    smiles_to_name = {}
    if args.names_csv:
        import pandas as pd
        df = pd.read_csv(args.names_csv)
        smiles_to_name = dict(zip(df["smiles"], df["name"]))

    labels = [smiles_to_name.get(s, s[:20] + "...") for s in smiles_list]

    # Pairwise cosine similarity
    norms_vec = norm(emb_matrix, axis=1, keepdims=True)
    norms_vec = np.where(norms_vec == 0, 1e-10, norms_vec)
    normed = emb_matrix / norms_vec
    sim_matrix = normed @ normed.T

    # Print full matrix
    print(f"\n=== Cosine Similarity Matrix ===")
    header = "".ljust(20) + "".join([l[:12].ljust(14) for l in labels])
    print(header)
    for i in range(n):
        row = labels[i][:20].ljust(20)
        for j in range(n):
            row += f"{sim_matrix[i, j]:.4f}".ljust(14)
        print(row)

    # Summary stats (off-diagonal only)
    mask = ~np.eye(n, dtype=bool)
    off_diag = sim_matrix[mask]
    print(f"\n=== Off-diagonal stats ===")
    print(f"  Mean:   {off_diag.mean():.4f}")
    print(f"  Std:    {off_diag.std():.4f}")
    print(f"  Min:    {off_diag.min():.4f}")
    print(f"  Max:    {off_diag.max():.4f}")
    print(f"  Median: {np.median(off_diag):.4f}")

    # Warnings
    print(f"\n=== Similarity Warnings (sim > {args.warn_threshold}) ===")
    found = False
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > args.warn_threshold:
                found = True
                print(f"WARNING - nearly identical embeddings:")
                print(f"  {labels[i]}")
                print(f"  {labels[j]}")
                print(f"  Cosine sim: {sim_matrix[i, j]:.4f}")
    if not found:
        print("None found.")

    # Plot
    if args.output_path:
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(sim_matrix, cmap="RdYlBu_r",
                        vmin=args.sim_floor, vmax=1.0)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if sim_matrix[i, j] > 0.9 else "black")

        plt.colorbar(im, ax=ax, label="Cosine Similarity")
        ax.set_title("3D Infomax Lipid Embedding Similarity")
        plt.tight_layout()
        plt.savefig(args.output_path, dpi=150)
        print(f"\nPlot saved to {args.output_path}")


if __name__ == "__main__":
    main()
