import os
import sys
import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from model import GNN


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atom_type = min(atomic_num - 1, 118)

        chirality = 0
        if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            chirality = 1
        elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            chirality = 2

        atom_features.append([atom_type, chirality])

    x = torch.tensor(atom_features, dtype=torch.long)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            bond_type = 0
        elif bt == Chem.BondType.DOUBLE:
            bond_type = 1
        elif bt == Chem.BondType.TRIPLE:
            bond_type = 2
        elif bt == Chem.BondType.AROMATIC:
            bond_type = 3
        else:
            bond_type = 0

        bd = bond.GetBondDir()
        if bd == Chem.BondDir.ENDDOWNRIGHT:
            bond_dir = 1
        elif bd == Chem.BondDir.ENDUPRIGHT:
            bond_dir = 2
        else:
            bond_dir = 0

        edge_index += [[i, j], [j, i]]
        edge_attr += [[bond_type, bond_dir], [bond_type, bond_dir]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def main():
    csv_path = r"C:\Users\danie\Local Desktop\lance_lipid_smiles.csv"
    df = pd.read_csv(csv_path)
    smiles_list = df['smiles'].tolist()

    weights_path = os.path.join(script_dir, 'supervised_contextpred.pth')
    print(f"Looking for weights at: {weights_path}")  # add this
    output_path = os.path.join(script_dir, 'lance_lipid_gin_embeddings.npy')

    model = GNN(num_layer=5, emb_dim=300, JK="last", drop_ratio=0.5, gnn_type="gin")
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    embeddings = {}
    failed = []

    for smiles in smiles_list:
        graph = smiles_to_graph(smiles)

        if graph is None:
            print(f"Failed to parse: {smiles}")
            failed.append(smiles)
            continue

        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)

        with torch.no_grad():
            node_repr = model(graph.x, graph.edge_index, graph.edge_attr)
            graph_repr = global_mean_pool(node_repr, graph.batch)

        embeddings[smiles] = graph_repr.squeeze().numpy()

    np.save(output_path, embeddings)
    print(f"Done. {len(embeddings)} embeddings saved, {len(failed)} failed.")
    if failed:
        print("Failed SMILES:")
        for s in failed:
            print(f"  {s}")


if __name__ == "__main__":
    main()