import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from numpy.linalg import norm

# Load name -> smiles mapping from CSV
csv_path = r'C:\Users\danie\Local Desktop\lance_lipid_smiles.csv'
df = pd.read_csv(csv_path)
smiles_to_name = dict(zip(df['smiles'], df['name']))

# Load embeddings
embeddings = np.load(
    r'C:\Users\danie\Local Desktop\COMET\gnn_encoding\GIN\lance_lipid_gin_embeddings.npy',
    allow_pickle=True
).item()

smiles_list = list(embeddings.keys())
emb_matrix = np.stack(list(embeddings.values()))
n = len(smiles_list)

# Use molecule names as labels, fall back to shortened SMILES if not found
labels = [smiles_to_name.get(s, s[:20] + '...') for s in smiles_list]

# Pairwise cosine similarity matrix
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = np.dot(emb_matrix[i], emb_matrix[j]) / (
            norm(emb_matrix[i]) * norm(emb_matrix[j])
        )

# Print warnings for nearly identical embeddings
print("=== Similarity Warnings (sim > 0.99) ===")
found = False
for i in range(n):
    for j in range(i+1, n):
        if sim_matrix[i, j] > 0.99:
            found = True
            print(f"WARNING - nearly identical embeddings:")
            print(f"  {labels[i]}")
            print(f"  {labels[j]}")
            print(f"  Cosine sim: {sim_matrix[i, j]:.4f}")
if not found:
    print("None found.")
print("Similarity check done\n")

# Plot similarity matrix
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(sim_matrix, cmap='plasma', vmin=0.8, vmax=1.0)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(labels, fontsize=9)

# Annotate cells
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                ha='center', va='center', fontsize=6,
                color='black' if sim_matrix[i, j] > 0.95 else 'white')

plt.colorbar(im, ax=ax, label='Cosine Similarity')
ax.set_title('Pairwise Cosine Similarity of GIN Lipid Embeddings', fontsize=13, pad=20)
plt.tight_layout()

output_path = r'C:\Users\danie\Local Desktop\COMET\gnn_encoding\GIN\gin_similarity_matrix.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()
print(f"Saved to {output_path}")