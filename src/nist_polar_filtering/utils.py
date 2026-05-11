"""
utils.py

This module provides utility functions for encoding SMILES strings,
generating molecular fingerprints, loading and processing data,
and generating molecular descriptors.
It is designed to be used with RDKit for molecular validation
and fingerprint generation, and Pandas for data manipulation.

Functions:
    encode_smiles(smiles): Encodes a list of SMILES strings into numerical format.
    generate_fingerprints(molecules): Generates fingerprints for a list of molecules.
    load_processed_data(input_file_name): Loads and processes the input data.
    generate_rdkit_descriptors(molecules): Generates descriptors for a list of molecules.

"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Descriptors import CalcMolDescriptors
from sklearn.preprocessing import StandardScaler
from tqdm.autonotebook import tqdm
from tqdm.contrib.concurrent import thread_map
from nist_polar_filtering import BASE_DIR
import re


def load_predicted_df(
    filename, model_names, offset: float = 0, **kwargs
) -> pd.DataFrame:
    """
    Load a dataframe containing predicted values from a file.

    Args:
        filename (str): The name of the file to load the dataframe from.
        model_names (List[str]): A list of model names to include in the dataframe.
        offset (float): The offset to use when calculating the mean threshold error.

    Returns:
        pd.DataFrame: A dataframe containing the predicted values.
    """
    dataframe = pd.read_csv(filename, sep=";", index_col=0)
    for model in model_names:
        if dataframe[model].dtype in [str, object]:
            dataframe[model] = [
                float(str(x).strip("[]")) for x in dataframe[model].to_numpy()
            ]
        dataframe[model] += offset
    dataframe["values"] += offset
    dataframe = dataframe.sort_values("smiles").reset_index(drop=True)
    return dataframe


def get_ecfp_from_smiles(smiles, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=1024)
    fp_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def get_maccs_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    return maccs_arr


def get_ecfp_from_inchi(inchi, radius: int = 2):
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return None
    fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=1024)
    fp_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def get_maccs_from_inchi(inchi):
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    return maccs_arr


def get_maccs_from_mol(mol):
    if not mol:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    return maccs_arr


def get_smiles_from_inchi(inchi):
    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return None
    else:
        return Chem.MolToSmiles(mol, isomericSmiles=False)


def get_smiles_dict(smiles):
    sym_dict = defaultdict(lambda: len(sym_dict) + 1)
    for s, smile in enumerate(tqdm(smiles)):
        if "i" in smile or "l" in smile or "r" in smile:
            smile = smile.replace("Si", "X").replace("Cl", "Y").replace("Br", "Z")
        for s2, sym in enumerate(smile):
            sym_dict[sym]
    return sym_dict


def encode_smiles(smiles, load_dict=None):
    """
    Encode a list of SMILES strings into numerical format.

    Parameters
    ----------
    smiles : list of str
        A list of SMILES strings to encode.

    Returns
    -------
    encoded_smiles : torch.Tensor
        A tensor of encoded SMILES strings.
    """
    if not load_dict:
        sym_dict = defaultdict(lambda: len(sym_dict) + 1)
    else:
        sym_dict = load_dict

    encoded_smiles = np.zeros((len(smiles), 256), dtype=np.int64)
    for s, smile in enumerate(tqdm(smiles)):
        if "i" in smile or "l" in smile or "r" in smile:
            smile = smile.replace("Si", "X").replace("Cl", "Y").replace("Br", "Z")
        for s2, sym in enumerate(smile):
            encoded_smiles[s, s2] = sym_dict[sym]
    return sym_dict, encoded_smiles


def generate_fingerprints(molecules):
    """
    Generate fingerprints for a list of molecules.

    Parameters
    ----------
    molecules : list of rdkit.rdChem.Mol
        A list of molecules to generate fingerprints for.

    Returns
    -------
    morgan_fingerprints : numpy.ndarray
        An array of Morgan fingerprints for the input molecules.
    rdkit_fingerprints : numpy.ndarray
        An array of RDKit fingerprints for the input molecules.
    """
    # FCFP and CatBoost
    print("Generating Morgan fingerprints")
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    morgan_fingerprints = np.array(
        thread_map(
            morgan_generator.GetCountFingerprintAsNumPy,
            molecules,
            chunksize=500,
            max_workers=8,
        )
    )

    # FCFP and CatBoost
    print("Generating RDKit fingerprints")
    rdkit_generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)
    rdkit_fingerprints = np.array(
        thread_map(
            rdkit_generator.GetCountFingerprintAsNumPy,
            molecules,
            chunksize=500,
            max_workers=8,
        )
    )
    return morgan_fingerprints, rdkit_fingerprints


def generate_rdkit_descriptors(molecules, load=False, prefix=""):
    """
    Generate descriptors for a list of molecules.

    Parameters
    ----------
    molecules : list of rdkit.rdChem.Mol
        A list of molecules to generate descriptors for.

    Returns
    -------
    descriptors : numpy.ndarray
        An array of RDKit descriptors for the input molecules.
    """
    print("Loading / Generating RDKit descriptors")
    if not load or not os.path.exists(
        BASE_DIR / f"data/processed/{prefix}_rdkit_descriptors.npy"
    ):
        descriptors = thread_map(
            CalcMolDescriptors, molecules, chunksize=500, max_workers=8
        )
        descriptors = pd.DataFrame(descriptors).fillna(value=0)
        print(descriptors.shape)
        np.save(
            BASE_DIR / f"data/processed/{prefix}_rdkit_descriptors.npy",
            descriptors.to_numpy(dtype=float, na_value=0),
        )
    else:
        descriptors = np.load(
            BASE_DIR / f"data/processed/{prefix}_rdkit_descriptors.npy"
        )

    print("Scaling RDKit descriptors to zero mean and unit variance")
    scl = StandardScaler()
    descriptors = scl.fit_transform(descriptors)
    return descriptors


mol_pattern = re.compile(r"(?:([A-Z][a-z]{0,1})(\d*))(?!\])")
isomol_pattern = re.compile(r"\[\d{1,3}([A-Z][a-z]{0,1})\](\d*)")


def get_form_dict(formula):
    if "[" in formula:
        formula_dict = dict(
            [
                (x, int(y)) if y else (x, 1)
                for (x, y) in re.findall(mol_pattern, formula)
            ]
        )
        iso_formula_findall = re.findall(isomol_pattern, formula)

        for val in iso_formula_findall:
            if formula_dict.get(val[0], None) is not None:
                formula_dict[val[0]] += int(val[1]) if val[1] else 1
            else:
                formula_dict[val[0]] = int(val[1]) if val[1] else 1
    else:
        formula_dict = dict(
            [
                (x, int(y)) if y else (x, 1)
                for (x, y) in re.findall(mol_pattern, formula)
            ]
        )
    return formula_dict


def get_form_list(formula):
    if "[" in formula:
        formula_list = [x for (x, _) in re.findall(mol_pattern, formula)]
        iso_formula_findall = re.findall(isomol_pattern, formula)

        for val in iso_formula_findall:
            if val[0] not in formula_list:
                formula_list.append(val[0])
    else:
        formula_list = [x for (x, _) in re.findall(mol_pattern, formula)]
    return formula_list


def get_clean_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    else:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
