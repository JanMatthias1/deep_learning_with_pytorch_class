import pandas as pd
import numpy as np
from unimol_tools import UniMolRepr

# 1. Read your SMILES list
csv_path = "../schnet/lance_lipid_smiles.csv" 
df = pd.read_csv(csv_path)
smiles_list = df['smiles'].tolist()

print(f"Loaded {len(smiles_list)} SMILES strings.")

# 2. Initialize the pre-trained UniMol model
print("Loading pre-trained UniMol model...")
unimol = UniMolRepr(data_type='molecule', remove_hs=False)

# 3. Extract the embeddings
print("Calculating conformations and extracting 512D embeddings...")
representations = unimol.get_repr(smiles_list, return_atomic_reprs=False)

# Safely extract the embeddings whether UniMol returns a list or a dict
if isinstance(representations, list):
    if isinstance(representations[0], dict):
        cls_embeddings = np.array([r['cls_repr'] for r in representations])
    else:
        cls_embeddings = np.array(representations)
else:
    cls_embeddings = np.array(representations['cls_repr'])

# 4. Format into a dictionary {SMILES: 512D_Array}
embeddings_dict = {}
for i, smiles in enumerate(smiles_list):
    embeddings_dict[smiles] = cls_embeddings[i]

# 5. Save to .npy
output_path = "unimol_lipid_embeddings.npy"
np.save(output_path, embeddings_dict)

print(f"\n✅ Successfully saved {len(embeddings_dict)} UniMol embeddings to {output_path}!")
print(f"Embedding dimension: {cls_embeddings.shape[1]}")