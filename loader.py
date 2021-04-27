import os
import torch
import pickle
from itertools import chain

import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from .util import (
    get_substructs,
    mol_to_graph_data_obj_simple,
    split_rdkit_mol_obj,
    get_largest_mol,
)


class MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset="zinc250k",
        partial_charge=False,
    ):
        """
        The main Dataset class for the project.

        Args:
            root (str): path to the root directory of the dataset.
            transform (callable): the on-the-fly data transformer.
            pre_transform (callable): the one-time data transformer for
                data preprocessing.
            prefilter (callable): the one-time filter for data preprocessing.
            dataset (str): name of the dataset.
            partial_charge (bool): use partial charge property.
        """
        self.dataset = dataset
        self.root = root
        self.partial_charge = partial_charge
        super(MoleculeDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        pass

    def add_data_to_list(self, smiles, mols, labels, data_list, data_smiles_list):
        for i in range(len(smiles)):
            rdkit_mol = mols[i]
            if rdkit_mol is None:
                continue
            data = mol_to_graph_data_obj_simple(rdkit_mol, self.partial_charge)
            data.id = torch.tensor([i])
            if len(labels.shape) > 1:
                data.y = torch.tensor(labels[i, :])
            else:
                data.y = torch.tensor([labels[i]])
            data.substructs = get_substructs(rdkit_mol)
            data_list.append(data)
            data_smiles_list.append(smiles[i])

    def load_zinc_standard_dataset(self, data_list, data_smiles_list):
        input_path = self.raw_paths[0]
        input_df = pd.read_csv(input_path, sep=",", compression="gzip", dtype="str")
        smiles_list = list(input_df["smiles"])
        zinc_id_list = list(input_df["zinc_id"])
        for i in range(len(smiles_list)):
            print(i, end="\r")
            s = smiles_list[i]
            # each example contains a single species
            rdkit_mol = AllChem.MolFromSmiles(s)
            if rdkit_mol is None:
                continue
            else:
                data = mol_to_graph_data_obj_simple(rdkit_mol, self.partial_charge)
                # add mol id
                id = int(zinc_id_list[i].split("ZINC")[1].lstrip("0"))
                data.id = torch.tensor([id])
                data.substructs = get_substructs(rdkit_mol)
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

    def load_chembl_dataset(self, data_list, data_smiles_list):
        (
            smiles_list,
            rdkit_mol_objs,
            folds,
            labels,
        ) = _load_chembl_with_labels_dataset(os.path.join(self.root, "raw"))
        for i in range(len(rdkit_mol_objs)):
            print(i, end="\r")
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol is not None:
                mw = Descriptors.MolWt(rdkit_mol)
                if 50 <= mw <= 900:
                    data = mol_to_graph_data_obj_simple(rdkit_mol, self.partial_charge)
                    # manually add mol id
                    data.id = torch.tensor([i])
                    data.substructs = get_substructs(rdkit_mol)
                    # No matches patterns
                    if len(data.substructs) == 0:
                        continue
                    data.y = torch.tensor(labels[i, :])
                    # fold information
                    if i in folds[0]:
                        data.fold = torch.tensor([0])
                    elif i in folds[1]:
                        data.fold = torch.tensor([1])
                    else:
                        data.fold = torch.tensor([2])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

    def load_dataset(self, data_list, data_smiles_list, method):
        smiles, mols, labels = method(self.raw_paths[0])
        self.add_data_to_list(smiles, mols, labels, data_list, data_smiles_list)

    def process(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == "zinc_standard_agent":
            self.load_zinc_standard_dataset(data_list, data_smiles_list)

        elif self.dataset == "chembl":
            self.load_chembl_dataset(data_list, data_smiles_list)

        elif self.dataset == "tox21":
            self.load_dataset(data_list, data_smiles_list, method=_load_tox21_dataset)

        elif self.dataset == "hiv":
            self.load_dataset(data_list, data_smiles_list, method=_load_hiv_dataset)

        elif self.dataset == "bace":
            self.load_dataset(data_list, data_smiles_list, method=_load_bace_dataset)

        elif self.dataset == "bbbp":
            self.load_dataset(data_list, data_smiles_list, method=_load_bbbp_dataset)

        elif self.dataset == "clintox":
            self.load_dataset(data_list, data_smiles_list, method=_load_clintox_dataset)

        elif self.dataset == "muv":
            self.load_dataset(data_list, data_smiles_list, method=_load_muv_dataset)

        elif self.dataset == "sider":
            self.load_dataset(data_list, data_smiles_list, method=_load_sider_dataset)

        elif self.dataset == "toxcast":
            self.load_dataset(data_list, data_smiles_list, method=_load_toxcast_dataset)
        else:
            raise ValueError("Invalid dataset name")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(
            os.path.join(self.processed_dir, "smiles.csv"), index=False, header=False
        )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# NB: only properly tested when dataset_1 is chembl_with_labels and dataset_2
# is pcba_pretrain
def merge_dataset_objs(dataset_1, dataset_2):
    """
    Naively merge 2 molecule dataset objects, and ignore identities of
    molecules. Assumes both datasets have multiple y labels, and will pad
    accordingly. ie if dataset_1 has obj_1 with y dim 1310 and dataset_2 has
    obj_2 with y dim 128, then the resulting obj_1 and obj_2 will have dim
    1438, where obj_1 have the last 128 cols with 0, and obj_2 have
    the first 1310 cols with 0.
    :return: pytorch geometric dataset obj, with the x, edge_attr, edge_index,
    new y attributes only
    """
    d_1_y_dim = dataset_1[0].y.size()[0]
    d_2_y_dim = dataset_2[0].y.size()[0]

    data_list = []
    # keep only x, edge_attr, edge_index, padded_y then append
    for d in dataset_1:
        old_y = d.y
        new_y = torch.cat([old_y, torch.zeros(d_2_y_dim, dtype=torch.long)])
        data_list.append(
            Data(x=d.x, edge_index=d.edge_index, edge_attr=d.edge_attr, y=new_y)
        )

    for d in dataset_2:
        old_y = d.y
        new_y = torch.cat([torch.zeros(d_1_y_dim, dtype=torch.long), old_y.long()])
        data_list.append(
            Data(x=d.x, edge_index=d.edge_index, edge_attr=d.edge_attr, y=new_y)
        )

    # create 'empty' dataset obj. Just randomly pick a dataset and root path
    # that has already been processed
    new_dataset = MoleculeDataset(
        root="dataset/chembl_with_labels", dataset="chembl_with_labels", empty=True
    )
    # collate manually
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)

    return new_dataset


def create_circular_fingerprint(mol, radius, size, chirality):
    """
    :param mol:
    :param radius:
    :param size:
    :param chirality:
    :return: np array of morgan fingerprint
    """
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=size, useChirality=chirality)
    return np.array(fp)


class MoleculeFingerprintDataset(data.Dataset):
    def __init__(self, root, dataset, radius, size, chirality=True):
        """
        Create dataset object containing list of dicts, where each dict
        contains the circular fingerprint of the molecule, label, id,
        and possibly precomputed fold information
        :param root: directory of the dataset, containing a raw and
        processed_fp dir. The raw dir should contain the file containing the
        smiles, and the processed_fp dir can either be empty or a
        previously processed file
        :param dataset: name of dataset. Currently only implemented for
        tox21, hiv, chembl_with_labels
        :param radius: radius of the circular fingerprints
        :param size: size of the folded fingerprint vector
        :param chirality: if True, fingerprint includes chirality information
        """
        self.dataset = dataset
        self.root = root
        self.radius = radius
        self.size = size
        self.chirality = chirality

        self._load()

    def _process(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == "chembl_with_labels":
            (
                smiles_list,
                rdkit_mol_objs,
                folds,
                labels,
            ) = _load_chembl_with_labels_dataset(os.path.join(self.root, "raw"))
            print("processing")
            for i in range(len(rdkit_mol_objs)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol is not None:
                    # # convert aromatic bonds to double bonds
                    fp_arr = create_circular_fingerprint(
                        rdkit_mol, self.radius, self.size, self.chirality
                    )
                    fp_arr = torch.tensor(fp_arr)
                    # manually add mol id
                    id = torch.tensor([i])  # id here is the index of the mol in
                    # the dataset
                    y = torch.tensor(labels[i, :])
                    # fold information
                    if i in folds[0]:
                        fold = torch.tensor([0])
                    elif i in folds[1]:
                        fold = torch.tensor([1])
                    else:
                        fold = torch.tensor([2])
                    data_list.append({"fp_arr": fp_arr, "id": id, "y": y, "fold": fold})
                    data_smiles_list.append(smiles_list[i])
        elif self.dataset == "tox21":
            smiles_list, rdkit_mol_objs, labels = _load_tox21_dataset(
                os.path.join(self.root, "raw/tox21.csv")
            )
            print("processing")
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # convert aromatic bonds to double bonds
                fp_arr = create_circular_fingerprint(
                    rdkit_mol, self.radius, self.size, self.chirality
                )
                fp_arr = torch.tensor(fp_arr)

                # manually add mol id
                id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                y = torch.tensor(labels[i, :])
                data_list.append({"fp_arr": fp_arr, "id": id, "y": y})
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == "hiv":
            smiles_list, rdkit_mol_objs, labels = _load_hiv_dataset(
                os.path.join(self.root, "raw/HIV.csv")
            )
            print("processing")
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                fp_arr = create_circular_fingerprint(
                    rdkit_mol, self.radius, self.size, self.chirality
                )
                fp_arr = torch.tensor(fp_arr)

                # manually add mol id
                id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                y = torch.tensor([labels[i]])
                data_list.append({"fp_arr": fp_arr, "id": id, "y": y})
                data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError("Invalid dataset name")

        # save processed data objects and smiles
        processed_dir = os.path.join(self.root, "processed_fp")
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(
            os.path.join(processed_dir, "smiles.csv"), index=False, header=False
        )
        with open(
            os.path.join(processed_dir, "fingerprint_data_processed.pkl"), "wb"
        ) as f:
            pickle.dump(data_list, f)

    def _load(self):
        processed_dir = os.path.join(self.root, "processed_fp")
        # check if saved file exist. If so, then load from save
        file_name_list = os.listdir(processed_dir)
        if "fingerprint_data_processed.pkl" in file_name_list:
            with open(
                os.path.join(processed_dir, "fingerprint_data_processed.pkl"), "rb"
            ) as f:
                self.data_list = pickle.load(f)
        # if no saved file exist, then perform processing steps, save then
        # reload
        else:
            self._process()
            self._load()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # if iterable class is passed, return dataset objection
        if hasattr(index, "__iter__"):
            dataset = MoleculeFingerprintDataset(
                self.root,
                self.dataset,
                self.radius,
                self.size,
                chirality=self.chirality,
            )
            dataset.data_list = [self.data_list[i] for i in index]
            return dataset
        else:
            return self.data_list[index]


def _load_tox21_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_hiv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df["HIV_active"]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_bace_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["mol"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df["Class"]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df["Model"]
    folds = folds.replace("Train", 0)  # 0 -> train
    folds = folds.replace("Valid", 1)  # 1 -> valid
    folds = folds.replace("Test", 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_bbbp_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [
        m if m is not None else None for m in rdkit_mol_objs_list
    ]
    preprocessed_smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None
        for m in preprocessed_rdkit_mol_objs_list
    ]
    labels = input_df["p_np"]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values


def _load_clintox_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [
        m if m is not None else None for m in rdkit_mol_objs_list
    ]
    preprocessed_smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None
        for m in preprocessed_rdkit_mol_objs_list
    ]
    tasks = ["FDA_APPROVED", "CT_TOX"]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values


def _load_muv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = [
        "MUV-466",
        "MUV-548",
        "MUV-600",
        "MUV-644",
        "MUV-652",
        "MUV-689",
        "MUV-692",
        "MUV-712",
        "MUV-713",
        "MUV-733",
        "MUV-737",
        "MUV-810",
        "MUV-832",
        "MUV-846",
        "MUV-852",
        "MUV-858",
        "MUV-859",
    ]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_sider_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_toxcast_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [
        m if m is not None else None for m in rdkit_mol_objs_list
    ]
    preprocessed_smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None
        for m in preprocessed_rdkit_mol_objs_list
    ]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values


def _load_chembl_with_labels_dataset(root_path):
    """
    Data from 'Large-scale comparison of machine learning methods for drug target
    prediction on ChEMBL'
    :param root_path: path to the folder containing the reduced chembl dataset
    :return: list of smiles, preprocessed rdkit mol obj list, list of np.array
    containing indices for each of the 3 folds, np.array containing the labels
    """
    # adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
    # first need to download the files and unzip:
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    # unzip and rename to chembl_with_labels
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced
    # /chembl20Smiles.pckl
    # into the dataPythonReduced directory
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced
    # /chembl20LSTM.pckl

    # 1. load folds and labels
    f = open(os.path.join(root_path, "folds0.pckl"), "rb")
    folds = pickle.load(f)
    f.close()

    f = open(os.path.join(root_path, "labelsHard.pckl"), "rb")
    targetMat = pickle.load(f)
    sampleAnnInd = pickle.load(f)
    targetAnnInd = pickle.load(f)
    f.close()

    targetMat = targetMat
    targetMat = targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd = targetAnnInd
    targetAnnInd = targetAnnInd - targetAnnInd.min()

    folds = [np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed = targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData = targetMat.A  # possible values are {-1, 0, 1}

    # 2. load structures
    f = open(os.path.join(root_path, "chembl20LSTM.pckl"), "rb")
    rdkitArr = pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print("preprocessing")
    for i in range(len(rdkitArr)):
        print(i, end="\r")
        m = rdkitArr[i]
        if m is None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None for m in preprocessed_rdkitArr
    ]  # bc some empty mol in the
    # rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData


def create_all_datasets():
    # create dataset
    downstream_dir = [
        "bace",
        "bbbp",
        "clintox",
        "hiv",
        "muv",
        "sider",
        "tox21",
        "toxcast",
    ]

    for dataset_name in downstream_dir:
        print(dataset_name)
        root = "contextSub/dataset/" + dataset_name
        dataset = MoleculeDataset(root, dataset=dataset_name, partial_charge=True)
        print(dataset)

    print("chembl")
    dataset = MoleculeDataset(
        root="contextSub/dataset/chembl", dataset="chembl", partial_charge=True
    )
    print(dataset)

    # print("zinc")
    # dataset = MoleculeDataset(
    #     root="contextSub/dataset/zinc_standard_agent", dataset="zinc_standard_agent"
    # )
    # print(dataset)


# test MoleculeDataset object
if __name__ == "__main__":
    create_all_datasets()
