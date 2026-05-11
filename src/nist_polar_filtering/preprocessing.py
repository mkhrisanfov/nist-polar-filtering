"""
preprocessing.py

This module contains functions for preprocessing METLIN SMRT molecular data from a CSV file.
It includes functions to filter entries based on retention time, validate InChI strings using RDKit,
and save the cleaned dataset to a new CSV file.
The module can be executed as a standalone script.

Functions:
    get_clean_dataset(filename):
        Filters entries based on retention time, validates InChI strings using RDKit,
        and saves the cleaned dataset.

Attributes:
    BASE_DIR (pathlib.Path): The base directory of the project.

"""

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import thread_map
from rdkit import Chem
from nist_polar_filtering import BASE_DIR
from nist_polar_filtering.utils import get_clean_smiles, get_form_dict, get_form_list
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm.auto import tqdm
from rdkit import rdBase


def load_processed_data(input_file_name):
    """
    Load and process the input data.

    Parameters
    ----------
    input_file_name : str
        The path to the input file.

    Returns
    -------
    input_inchi : numpy.ndarray
        The InChI for the input molecules.
    input_molecules : numpy.ndarray
        The input molecules.
    input_retention_values : numpy.ndarray
        The input retention times.
    """
    input_df = pd.read_csv(input_file_name, index_col=0)
    input_inchi = input_df["inchi"].to_numpy()
    input_retention_values = input_df["ri"].to_numpy() / 1000
    print("Generating molecules")
    input_molecules = np.array(
        thread_map(Chem.MolFromInchi, input_inchi, chunksize=500)
    )
    return input_inchi, input_molecules, input_retention_values


def get_clean_dataset(filename):
    ri_df = pd.read_csv(filename, sep="|", index_col=0)
    print(len(ri_df))

    print("Excluding problematic entries")
    ri_df = ri_df.loc[ri_df["problems"].isna()]
    print(len(ri_df))
    print("Excluding retention indices other than Kovats and Linear")
    ri_df = ri_df.loc[
        [
            True if x in ["Linear RI", "Normal alkane RI", "Kovats RI"] else False
            for x in ri_df["ri type"].values
        ]
    ]
    print(len(ri_df))
    print("Excluding columns other than Standard Non-polar and Semi-Standard Non-polar")
    ri_df = ri_df.loc[
        (ri_df["phase class"] == "Semi-standard non-polar")
        | (ri_df["phase class"] == "Standard non-polar")
    ]
    print(len(ri_df))

    print("Excluding all entries with uncommon atoms")
    atomlist = []
    for x in tqdm(ri_df["formula"], leave=False):
        atomlist.extend(get_form_dict(x).keys())

    allowed_atoms = {
        k: v
        for (k, v) in sorted(
            zip(*np.unique(atomlist, return_counts=True)), key=lambda x: -x[1]
        )
        if v > 1000
    }
    allowed_atoms = list(allowed_atoms.keys())
    allowed_atoms.append("I")
    print(allowed_atoms)

    allowed_atoms_mask = [
        np.all(np.isin(get_form_list(x), allowed_atoms))
        for x in tqdm(ri_df["formula"].values, leave=False)
    ]
    ri_df = ri_df.loc[allowed_atoms_mask]
    print(len(ri_df))

    print("Excluding smiles longer than 256 symbols")
    smiles_mask = [
        True if len(x) < 256 else False
        for x in tqdm(ri_df["smiles"].values, leave=False)
    ]
    ri_df = ri_df.loc[smiles_mask]
    print(len(ri_df))

    print("Excluding all broken molecules")
    molecules = list(map(Chem.MolFromSmiles, tqdm(ri_df["smiles"].values, leave=False)))
    molecules_mask = [True if mol is not None else False for mol in molecules]
    molecules = np.array(molecules)[molecules_mask]
    ri_df = ri_df.loc[molecules_mask]
    print(len(ri_df))

    print("Obtaining non-isomeric smiles for train-eval split")
    noniso_smiles = list(
        map(
            lambda x: Chem.MolToSmiles(x, isomericSmiles=False, canonical=True),
            tqdm(molecules, leave=False),
        )
    )

    return (
        molecules,
        ri_df["smiles"].to_numpy(),
        noniso_smiles,
        ri_df["ri"].to_numpy() / 1000,
        ri_df["phase name"].str.lower(),
    )


def load_nist_2023_dataset(filename: str):
    """
    Load and filter the NIST RI 2023 database from a TSV file.

    Reads the tab-separated NIST 23 retention index database, filters entries
    to include only Kovats/Linear RI types, capillary/packed columns, valid
    molecules with allowed atoms (C, H, O, N, Cl, Si, F, S, Br, P, I), and
    SMILES strings shorter than 256 characters.

    Parameters
    ----------
    filename : str
        Path to the NIST RI 2023 TSV file.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Filtered retention index dataframe and corresponding RDKit molecule objects.
    """
    ri_df = pd.read_csv(filename, sep="\t", index_col=0)
    df_len = len(ri_df)
    print(f"Initial DB size: {df_len}")

    COLUMN_TYPES = ["Capillary", "Packed"]
    DATA_TYPES = ["Normal alkane RI", "Linear RI", "Kovats RI"]

    ri_df = ri_df.loc[ri_df["data_type"].isin(DATA_TYPES)]
    print(f"{df_len - len(ri_df)} non-Kovats and non-Linear indices excluded")
    df_len = len(ri_df)
    print(f"DB size: {df_len}")

    ri_df = ri_df.loc[ri_df["column_type"].isin(COLUMN_TYPES)]
    print(
        f"{df_len - len(ri_df)} entries with "
        "columns types other than Capillary and Packed excluded"
    )
    df_len = len(ri_df)
    print(f"DB size: {df_len}")

    smiles_mask = [
        True if len(x) < 256 else False
        for x in tqdm(ri_df["smiles"].values, leave=False)
    ]
    ri_df = ri_df.loc[smiles_mask]
    print(f"{df_len - len(ri_df)} SMILES longer than 256 symbols excluded")
    df_len = len(ri_df)
    print(f"DB size: {df_len}")

    print("Generate molecules")
    rdBase.DisableLog("rdApp.*")
    molecules = list(map(Chem.MolFromInchi, tqdm(ri_df["inchi"].values, leave=False)))
    molecules_mask = [True if mol is not None else False for mol in molecules]
    molecules = np.array(molecules)[molecules_mask]
    ri_df = ri_df.loc[molecules_mask]
    rdBase.EnableLog("rdApp.*")
    print(f"{df_len - len(ri_df)} broken molecules excluded")
    df_len = len(ri_df)
    print(f"DB size: {df_len}")

    formulas = list(map(CalcMolFormula, tqdm(molecules, leave=False)))
    allowed_atoms = ["C", "H", "O", "N", "Cl", "Si", "F", "S", "Br", "P", "I"]
    print("Allowed atoms")
    print(allowed_atoms)

    allowed_atoms_mask = [
        np.all(np.isin(get_form_list(x), allowed_atoms))
        for x in tqdm(formulas, leave=False)
    ]
    ri_df = ri_df.loc[allowed_atoms_mask]
    molecules = molecules[allowed_atoms_mask]
    print(f"{df_len - len(ri_df)} entries with uncommon atoms excluded")

    ri_df["ri"] = ri_df["value"].str.rstrip(" iu").to_numpy(dtype=float) / 1000
    print(f"Final DB size: {len(ri_df)}")

    return ri_df, molecules
