"""
Pairwise cosine similarity check for SchNet lipid embeddings.

Adapted from the GIN similarity_check.py.
SchNet embeddings are 128-dim (vs GIN's 300-dim), otherwise identical.

Usage:
  python similarity_check_schnet.py --embeddings_path schnet_embeddings.npy
  python similarity_check_schnet.py --embeddings_path schnet_embeddings.npy --names_csv lance_lipid_smiles.csv
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


def main():
    parser = argparse.ArgumentParser(
        description='Cosine similarity matrix for SchNet embeddings'
    )
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to .npy embeddings dict')
    parser.add_argument('--names_csv', type=str, default=None,
                        help='Optional CSV with smiles,name columns for labels')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for similarity matrix plot')
    parser.add_argument('--sim_floor', type=float, default=0.8,
                        help='Lower bound for colormap (default: 0.8)')
    parser.add_argument('--warn_threshold', type=float, default=0.99,
                        help='Threshold for near-identical warning (default: 0.99)')
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
        smiles_to_name = dict(zip(df['smiles'], df['name']))

    labels = [smiles_to_name.get(s, s[:20] + '...') for s in smiles_list]

    # Pairwise cosine similarity
    norms_vec = norm(emb_matrix, axis=1, keepdims=True)
    norms_vec = np.where(norms_vec == 0, 1e-10, norms_vec)
    normed = emb_matrix / norms_vec
    sim_matrix = normed @ normed.T

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
    print("Similarity check done\n")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim_matrix, cmap='plasma', vmin=args.sim_floor, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=6,
                    color='black' if sim_matrix[i, j] > 0.95 else 'white')

    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    ax.set_title('Pairwise Cosine Similarity of SchNet Lipid Embeddings',
                 fontsize=13, pad=20)
    plt.tight_layout()

    output = args.output_path
    if output is None:
        output = args.embeddings_path.replace('.npy', '_similarity_matrix.png')
    plt.savefig(output, dpi=600, bbox_inches='tight')
    print(f"Saved to {output}")
    plt.show()


if __name__ == "__main__":
    main()
