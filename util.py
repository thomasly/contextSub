import os.path as osp
import json

import torch
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Batch
import matplotlib.pyplot as plt

from .model import GNN


allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


def check_same_molecules(s1, s2):
    mol1 = AllChem.MolFromSmiles(s1)
    mol2 = AllChem.MolFromSmiles(s2)
    return AllChem.MolToInchi(mol1) == AllChem.MolToInchi(mol2)


def check_smiles_validity(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m:
        return True
    else:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split(".")
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_substruct_x(substruct, mol, partial_charge):
    atom_features_list = []
    for atom_idx in substruct:
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_feature = [
            allowable_features["possible_atomic_num_list"].index(atom.GetAtomicNum())
        ] + [allowable_features["possible_chirality_list"].index(atom.GetChiralTag())]
        if partial_charge:
            atom_feature += [atom.GetDoubleProp("_GasteigerCharge")]
        atom_features_list.append(atom_feature)
    if partial_charge:
        x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    else:
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    return x


def get_substruct_bonds(substruct, mol):
    """ Get the bonds within a substruct
    """
    atom_pairs = list()
    for i in substruct:
        for j in substruct:
            if j > i:
                atom_pairs.append((i, j))
    bonds = list()
    for pair in atom_pairs:
        bond = mol.GetBondBetweenAtoms(*pair)
        if bond is not None:
            bonds.append(bond)
    return bonds


def get_substruct_edge_attrs(substruct, mol, starting_idx=0):
    num_bond_features = 2  # bond type, bond direction
    bonds = get_substruct_bonds(substruct, mol)
    idx_map = dict()
    for i, atom_idx in enumerate(substruct):
        idx_map[atom_idx] = i + starting_idx
    if len(bonds) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in bonds:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features["possible_bonds"].index(bond.GetBondType())
            ] + [allowable_features["possible_bond_dirs"].index(bond.GetBondDir())]
            edges_list.append((idx_map[i], idx_map[j]))
            edge_features_list.append(edge_feature)
            edges_list.append((idx_map[j], idx_map[i]))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    return edge_index, edge_attr


def mol_to_graph_data_obj_simple(
    mol, partial_charge=False, substruct_input=False, pattern_path=None
):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices.

    Args:
        mol: rdkit mol object.
        partial_charge (bool): if to add atom partial charge as atom property.
        substruct_input (bool): add substructure nodes into data.x
        patten_path (str): path to the csv file with PubChem SMARTS patterns
    Returns:
        graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms, num_atom_features = 2
    atom_features_list = []
    if partial_charge:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        for atom in mol.GetAtoms():
            if np.isnan(atom.GetDoubleProp("_GasteigerCharge")):
                return
    for atom in mol.GetAtoms():
        atom_feature = [
            allowable_features["possible_atomic_num_list"].index(atom.GetAtomicNum())
        ] + [allowable_features["possible_chirality_list"].index(atom.GetChiralTag())]
        if partial_charge:
            atom_feature += [atom.GetDoubleProp("_GasteigerCharge")]
        atom_features_list.append(atom_feature)
    if partial_charge:
        x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    else:
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features["possible_bonds"].index(bond.GetBondType())
            ] + [allowable_features["possible_bond_dirs"].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.substructs = get_substructs(mol, pattern_path)
    starting_idx = len(list(mol.GetAtoms()))
    if substruct_input:
        for patterns in data.substructs:
            for substruct in patterns:
                substruct_x = get_substruct_x(substruct, mol, partial_charge)
                substruct_edge_list, substruct_edge_attrs = get_substruct_edge_attrs(
                    substruct, mol, starting_idx=starting_idx
                )
                data.x = torch.cat([data.x, substruct_x], 0)
                data.edge_index = torch.cat([data.edge_index, substruct_edge_list], 1)
                data.edge_attr = torch.cat([data.edge_attr, substruct_edge_attrs], 0)
                starting_idx += len(substruct)
    return data


def graph_data_obj_to_mol_simple(
    data_x, data_edge_index, data_edge_attr, partial_charge=False
):
    """
    Convert pytorch geometric data obj to rdkit mol object. NB: Uses simplified
    atom and bond features, and represent as indices.
    :param: data_x:
    :param: data_edge_index:
    :param: data_edge_attr
    :return:
    """
    mol = Chem.RWMol()

    # atoms
    atom_features = data_x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        if partial_charge:
            atomic_num_idx, chirality_tag_idx, pc = atom_features[i]
        else:
            atomic_num_idx, chirality_tag_idx = atom_features[i, :2]
        atomic_num = allowable_features["possible_atomic_num_list"][atomic_num_idx]
        chirality_tag = allowable_features["possible_chirality_list"][chirality_tag_idx]
        atom = Chem.Atom(atomic_num)
        atom.SetChiralTag(chirality_tag)
        if partial_charge:
            atom.SetDoubleProp("_GasteigerCharge", pc)
        mol.AddAtom(atom)

    # bonds
    edge_index = data_edge_index.cpu().numpy()
    edge_attr = data_edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        bond_type = allowable_features["possible_bonds"][bond_type_idx]
        bond_dir = allowable_features["possible_bond_dirs"][bond_dir_idx]
        mol.AddBond(begin_idx, end_idx, bond_type)
        # set bond direction
        new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        new_bond.SetBondDir(bond_dir)

    # Chem.SanitizeMol(mol) # fails for COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(
    # C)C(OC)=C2C)C=C1, when aromatic bond is possible
    # when we do not have aromatic bonds
    # Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    return mol


def graph_data_obj_to_nx_simple(data, partial_charge=False):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        if partial_charge:
            atomic_num_idx, chirality_tag_idx, pc = atom_features[i]
            G.add_node(
                i,
                atom_num_idx=atomic_num_idx,
                chirality_tag_idx=chirality_tag_idx,
                partial_charge=pc,
            )
        else:
            atomic_num_idx, chirality_tag_idx = atom_features[i, :2]
            G.add_node(
                i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx
            )

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(
                begin_idx,
                end_idx,
                bond_type_idx=bond_type_idx,
                bond_dir_idx=bond_dir_idx,
            )

    return G


def nx_to_graph_data_obj_simple(G, partial_charge=False):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # atoms, num_atom_features = 2, (atom type, chirality tag)
    atom_features_list = []
    for _, node in G.nodes(data=True):
        if partial_charge:
            atom_feature = [
                node["atom_num_idx"],
                node["chirality_tag_idx"],
                node["partial_charge"],
            ]
        else:
            atom_feature = [node["atom_num_idx"], node["chirality_tag_idx"]]
        atom_features_list.append(atom_feature)
    if partial_charge:
        x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    else:
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge["bond_type_idx"], edge["bond_dir_idx"]]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    :param mol: rdkit mol object
    :param n_iter: number of iterations. Default 12
    :return: list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(
        mol, nIter=n_iter, throwOnParamFailure=True
    )
    partial_charges = [float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """

    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(
            AllChem.MolFromSmiles(smiles), isomericSmiles=False
        )
        mol = AllChem.MolFromSmiles(smiles)
        if mol is not None:
            if "." in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


def get_substructs(mol, pattern_path=None):
    """ Get substructures from a mol.

    Args:
        mol (RDKit Mol): the Molecule object from which to find the substructures.
        pattern_path (str): path to the patterns file. Default is "contextSub/resources/
            pubchemFPKeys_to_SMARTSpattern.csv"

    Returns:
        substructs (list): list of lists of atom indices belonging to each substrucure.
    """
    if pattern_path is None:
        pattern_path = osp.join(
            "contextSub", "resources", "pubchemFPKeys_to_SMARTSpattern.csv"
        )
    patterns_df = pd.read_csv(pattern_path)
    patterns = [Chem.MolFromSmarts(sm) for sm in patterns_df.SMARTS]
    substructs = list()
    for pat in patterns:
        matches = mol.GetSubstructMatches(pat)
        if len(matches) > 0:
            substructs.append(matches)
    return substructs


def _load_candidates(pattern, chemicals, partial_charge=False):
    candidates = json.load(open(chemicals))
    data_list = []
    y = list()
    label = 0
    for key, sms in candidates.items():
        y.extend([label] * len(sms))
        label += 1
        for sm in sms:
            mol = Chem.MolFromSmiles(sm)
            data = mol_to_graph_data_obj_simple(mol, partial_charge=partial_charge)
            data.atom = str(key)
            data.substructs = torch.tensor(
                mol.GetSubstructMatch(pattern), dtype=torch.int
            )
            data_list.append(data)
    return data_list, y


def _get_slices(batch):
    slices = list()
    n = 0
    while n < batch.batch.size(0):
        start = n
        curr_value = batch.batch[n].item()
        while n < batch.batch.size(0) and batch.batch[n].item() == curr_value:
            n += 1
        slices.append(slice(start, n))
    return slices


def evaluate_pretraining(pattern, chemicals, model_path, partial_charge=False):
    """ Evaluate the pretraining by analyzing the embeddings of the same substructure
    within different contexts.

    Args:
        pattern (RDKit Mol): the pattern to decide the centeral substructure.
        chemicals (str): path to the json file with chemicals.
        model_path (str): path to the pretrained model.
        partial_charge (bool): the model takes partial charge as atom property.

    Returns:
        pattern_embs: the embeddings of patterns.
        y: pattern labels based on input.
    """
    data_list, y = _load_candidates(pattern, chemicals, partial_charge)
    batch = Batch.from_data_list(data_list)
    slices = _get_slices(batch)
    model = GNN(
        num_layer=5,
        emb_dim=300,
        JK="last",
        drop_ratio=0.5,
        gnn_type="gin",
        partial_charge=partial_charge,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    embeddings = model(batch)
    molecule_embs = [embeddings[sl] for sl in slices]
    pattern_embs = list()
    for data, emb in zip(data_list, molecule_embs):
        pattern_embs.append(
            torch.mean(emb[data.substructs.to(torch.long)], 0).detach().numpy()
        )
    return pattern_embs, y


def plot_embedding(X, y, title=None, mode="text", cmap=None):
    if cmap is None:
        cmap = ["red", "green", "blue", "orange", "magenta", "gray"]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    _ = plt.subplot(111)
    if mode == "text":
        for i in range(X.shape[0]):
            plt.text(
                X[i, 0],
                X[i, 1],
                f"{y[i]}.{i}",
                color=cmap[y[i]],
                fontdict={"weight": "bold", "size": 9},
            )
    elif mode == "dot":
        for i in range(X.shape[0]):
            plt.scatter(X[i, 0], X[i, 1], color=cmap[y[i]])
    else:
        raise ValueError(f"Wrong mode: {mode}")
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
