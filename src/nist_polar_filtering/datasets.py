"""
datasets.py
-----------

This module contains classes and functions for creating and handling
different types of datasets used in the filtering metlin project. These datasets
are primarily used for training and evaluating various machine learning models
in the context of HPLC retention time prediction.

Classes:
    CNN_Dataset: A custom Dataset class for handling encoded SMILES strings
                and their corresponding retention times.
    FCD_Dataset: A custom Dataset class for handling molecular descriptors
                and their corresponding retention times.
    FCFP_Dataset: A custom Dataset class for handling molecular fingerprints
                (ECFP and RDKit) and their corresponding retention times.
    GNNIMDataset: A PyTorch Geometric InMemoryDataset for graph-structured data.

Functions:
    get_dict_gnn_dataset: Create a dictionary-based GNN dataset.
    prepare_gnn_dataset: Prepare graph data for GNN training.
    get_gnn_dataset: Create a graph-structured dataset for GNN models.
"""

from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
from torch_geometric.data import Data

# from torch_geometric.data.dataset import Dataset as GDataset
from torch_geometric.data import InMemoryDataset

# import copy


class CNN_Dataset(Dataset):
    def __init__(self, encoded_smis, encoded_phases, retention_values):
        self.encoded_smis = torch.tensor(encoded_smis, dtype=torch.long)
        self.encoded_phases = torch.tensor(encoded_phases, dtype=torch.long)
        self.retention_values = torch.FloatTensor(retention_values)

    def __getitem__(self, index):
        return (
            self.encoded_smis[index],
            self.encoded_phases[index],
            self.retention_values[index],
        )

    def __len__(self):
        return len(self.retention_values)


class FCD_Dataset(Dataset):
    def __init__(self, descriptors, phases, retention_values):
        retention_values = np.vstack(retention_values)
        self.retention_values = torch.FloatTensor(retention_values)
        self.descriptors = torch.FloatTensor(descriptors)
        self.phases = torch.LongTensor(phases)

    def __getitem__(self, index):
        return (
            self.descriptors[index],
            self.phases[index],
            self.retention_values[index],
        )

    def __len__(self):
        return len(self.retention_values)


class FCFP_Dataset(Dataset):
    def __init__(
        self, rdkit_fingerprints, morgan_fingerprints, phases, retention_values
    ):
        retention_values = np.vstack(retention_values)
        self.retention_values = torch.FloatTensor(retention_values)
        self.rdkit_fingerprints = torch.FloatTensor(np.vstack(rdkit_fingerprints))
        self.morgan_fingerprints = torch.FloatTensor(np.vstack(morgan_fingerprints))
        self.phases = torch.LongTensor(phases)

    def __getitem__(self, index):
        return (
            self.rdkit_fingerprints[index],
            self.morgan_fingerprints[index],
            self.phases[index],
            self.retention_values[index],
        )

    def __len__(self):
        return len(self.retention_values)


class GNNIMDataset(InMemoryDataset):
    """
    A PyTorch Geometric InMemoryDataset class for handling graph data.

    Parameters:
        raw_data (list of dict): Raw data containing molecular graphs and their properties.

    Attributes:
        data (Data): PyG Data object containing the processed dataset.
        slices (dict): Slices information for the processed dataset.
    """

    def __init__(self, raw_data):
        super().__init__(None, None, None)
        self.data, self.slices = self.process_data(raw_data)

    @property
    def raw_file_names(self):
        return ["raw_data.pt"]  # Expecting raw data file

    @property
    def processed_file_names(self):
        return ["data.pt"]  # Processed dataset file

    def process_data(self, raw_data):
        # Load raw data (list of dicts)
        data_list = []  # List of `Data` objects

        for sample in raw_data:
            # Convert features to tensors
            # Fingerprints as node features
            x = torch.tensor(sample["fingerprints"], dtype=torch.long)
            edge_index = torch.tensor(
                sample["mol_bonds"], dtype=torch.long
            )  # Connectivity
            y = torch.tensor(
                [sample["retention_time"]], dtype=torch.float
            )  # Retention time
            phase = torch.tensor([sample["phases"]], dtype=torch.long)

            # Create PyG `Data` object
            data = Data(x=x, edge_index=edge_index, phase=phase, y=y)

            data_list.append(data)

        # Convert list of Data objects into a single `Data` object
        return self.collate(data_list)


def get_dict_gnn_dataset(fingerprints, mol_bonds, molecular_properties, phases):
    """
    Converts fingerprints, molecular bonds, and molecular properties into a list of dictionaries.

    Parameters:
        fingerprints (list of lists): RDKit fingerprints.
        mol_bonds (list of lists): Molecular bond indices.
        molecular_properties (list of float): molecular properties corresponding to the molecules.

    Returns:
        list of dict: List of dictionaries containing molecular data.
    """
    dataset = []
    for i in tqdm(range(len(molecular_properties)), leave=False):
        new_mol_dict = {}
        new_mol_dict["fingerprints"] = fingerprints[i]
        new_mol_dict["mol_bonds"] = mol_bonds[i]
        new_mol_dict["retention_time"] = molecular_properties[i]
        new_mol_dict["phases"] = phases[i]
        dataset.append(new_mol_dict)
    return dataset


def prepare_gnn_dataset(molecules, dicts=None):
    """
    Prepares graph data for GNN training by extracting atom and bond features,
    and creating fingerprints.

    Parameters:
        molecules (list of RDKit Mol): List of RDKit molecule objects.

    Returns:
        tuple: A tuple containing the number of unique fingerprints, a list of fingerprints,
               and a list of molecular bonds.
    """
    if not dicts:
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict) + 1)
    else:
        fingerprint_dict = dicts

    all_fingerprints = []
    all_mol_bonds = []
    for m, mol in enumerate(tqdm(molecules, leave=False)):
        fingerprints = np.zeros(mol.GetNumAtoms(), dtype=np.int64)
        for a, atom in enumerate(mol.GetAtoms()):
            atom_str = "-".join(
                map(
                    str,
                    [
                        atom.GetSmarts(),
                        atom.GetTotalDegree(),
                        atom.GetTotalNumHs(),
                        atom.GetFormalCharge(),
                    ],
                )
            )
            neighbors_str = "".join(
                sorted([ngh.GetSmarts() for ngh in atom.GetNeighbors()])
            )
            fingerprints[a] = fingerprint_dict[f"{atom_str}_{neighbors_str}"]
        mol_bonds = np.zeros((2, 2 * mol.GetNumBonds()), dtype=np.int64)
        for b, bond in enumerate(mol.GetBonds()):
            mol_bonds[0, b], mol_bonds[1, b] = (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
            )
            mol_bonds[0, -b - 1], mol_bonds[1, -b - 1] = (
                bond.GetEndAtomIdx(),
                bond.GetBeginAtomIdx(),
            )

        all_fingerprints.append(fingerprints)
        all_mol_bonds.append(mol_bonds)
    print("FPs Dict", len(fingerprint_dict))
    num_fingerprints = len(fingerprint_dict) + 1
    return num_fingerprints, all_fingerprints, all_mol_bonds, fingerprint_dict
