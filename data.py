# data.py
"""
Data loading and feature engineering:
- Novamag dataset: X, y
- Materials Project dataset: mp_X, mp_y
- Generic stoichiometric array builder
- Generic dataset splitting helper
"""

import re
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import alloys as al
import mp_alloys as mp_al  # only used in the MP case study


# ====== Shared utilities: load element tables ======

def load_periodic_tables(
    pt_path: str = "./data/Periodic-table/periodic_table.xlsx",
    mm_path: str = "./data/Miedema-model/Miedema-model-reduced.xlsx",
):
    """Load the periodic table PT and the Miedema model MM."""
    PT = al.importPT(pt_path)
    MM = al.importMM(mm_path)
    return PT, MM


# ====== Novamag section ======

def load_novamag_raw(novamag_dir: str = "./data/Novamag_Data_Files") -> pd.DataFrame:
    """
    Import the raw Novamag data, keeping only chemical formula and saturation magnetisation.
    Matches notebook cells 7, 8, 9, and 10.
    """
    X = al.importNovamag(novamag_dir)

    print(f"The total number of imported features is {len(X.columns)}")

    # Fix the 'none' values
    X = X.fillna(value=np.nan).copy()

    # Find columns with missing values
    na_cols = [col for col in X.columns if X[col].isna().any()]
    print(
        "The number of features with at least one NaN value is "
        f"{len(na_cols)}"
    )

    # Drop colums with more than 10 NaN values
    dropped_cols = []
    for col in na_cols:
        count = X[col].isna().sum()
        print(f"Column '{col}' has {count} nan values")
        if count > 10:
            print(f"Dropping column '{col}'")
            dropped_cols.append(col)
            X = X.drop(col, axis=1).copy()
    print(f"Number of dropped features is: {len(dropped_cols)}")

    # Drop all columns except target and chemical formula
    X = X[["chemical formula", "saturation magnetization"]]
    return X


def build_novamag_features(
    X_raw: pd.DataFrame,
    PT: pd.DataFrame,
    MM: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Engineer features for Novamag and return X_feat, y, and the feature list.
    Matches notebook cells 18, 36, 38, and 40.
    """
    X = X_raw.copy()

    # compoundradix: number of species in the compound (binary, ternary, etc.)
    X["compoundradix"] = al.get_CompoundRadix(PT, X)

    # stoichiometric array
    stoich_array = al.get_stoich_array(X, PT)

    # Compute element-weighted features from the stoichiometric array
    X["stoicentw"] = al.get_StoicEntw(stoich_array)          # mixing entropy
    X["Zw"] = al.get_Zw(PT, stoich_array)                     # atomic weight
    # X['compoundradix'] was computed above
    X["periodw"] = al.get_Periodw(PT, stoich_array)           # period
    X["groupw"] = al.get_Groupw(PT, stoich_array)             # group
    X["meltingTw"] = al.get_MeltingTw(PT, stoich_array)       # melting point
    X["miedemaH"] = al.get_Miedemaw(MM, stoich_array)         # Miedema mixing enthalpy
    X["valencew"] = al.get_Valencew(PT, stoich_array)         # valence
    X["electronegw"] = al.get_Electronegw(PT, stoich_array)   # electronegativity

    # Feature list kept consistent with the notebook
    my_cols = [
        "stoicentw",
        "Zw",
        "periodw",
        "groupw",
        "meltingTw",
        "miedemaH",
        "valencew",
        "electronegw",
    ]

    # Remove rows with missing target, separate target from predictors
    X.dropna(
        axis=0,
        subset=["saturation magnetization"] + my_cols,
        inplace=True,
    )

    # Drop alloys which are below magnetic cutoff (determined through prior model optimisation)
    X.drop(X[X["saturation magnetization"] < 0.18].index, axis=0, inplace=True)

    # Round the saturation magnetization to 1.d.p
    X["saturation magnetization"] = pd.to_numeric(
        X["saturation magnetization"]
    ).round(decimals=2)

    # Group duplicates by chemical formula and replace values with median
    X = X.groupby(by="chemical formula").median()
    X.index = range(len(X))

    # Define target and drop from X
    y = X["saturation magnetization"]
    X.drop(["saturation magnetization"], axis=1, inplace=True)

    # Keep only the feature columns while preserving the my_cols order
    X = X[my_cols].copy()

    return X, y, my_cols


def load_novamag_dataset(
    novamag_dir: str = "./data/Novamag_Data_Files",
    pt_path: str = "./data/Periodic-table/periodic_table.xlsx",
    mm_path: str = "./data/Miedema-model/Miedema-model-reduced.xlsx",
):
    """
    Convenience wrapper for Novamag: load the data and engineer the features.
    """
    PT, MM = load_periodic_tables(pt_path, mm_path)
    X_raw = load_novamag_raw(novamag_dir)
    X, y, my_cols = build_novamag_features(X_raw, PT, MM)
    return X, y, my_cols, PT, MM


# ====== Materials Project section ======

def load_mp_raw(csv_path: str) -> pd.DataFrame:
    """
    Read the Materials Project CSV export.
    Matches notebook cells 21 and 23.
    """
    Y = pd.read_csv(csv_path)

    # Keep only magnetic systems
    Y = Y[Y["is_magnetic"]]

    # Remove entries containing rare-earth elements
    rare_earth_elements = [
        "La",
        "Sc",
        "Dy",
        "Sm",
        "Lu",
        "Er",
        "Y",
        "Pr",
        "Nd",
        "Gd",
        "Tm",
        "Pm",
        "Ce",
        "Tb",
        "Eu",
        "Ho",
        "Yb",
    ]
    for element in rare_earth_elements:
        Y = Y[~Y["composition_reduced"].str.contains(element)]

    # Remove uncommon or commercially unavailable elements
    non_commercial_elements = [
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
    ]
    for element in non_commercial_elements:
        Y = Y[~Y["composition_reduced"].str.contains(element)]

    return Y


# ====== Generic stoichiometric array builder (used for MP and case studies) ======

# Note: mirrors the logic from notebook cell 45

pattern = r"([A-Z][a-z]*)(\d*)"


def extract_elements(compound: str):
    """Extract elements and their counts from a formula string."""
    element_counts = {}
    matches = re.findall(pattern, compound)
    for match in matches:
        element = match[0]
        count = int(match[1]) if match[1] else 1
        element_counts[element] = count
    return element_counts


def build_stoichiometric_array(composition_column: pd.Series) -> pd.DataFrame:
    """
    Build the stoichiometric array for a composition column.
    Matches notebook cell 45.
    """
    # Extract unique element symbols from all compositions
    all_elements = set()
    for compound in composition_column:
        all_elements.update(extract_elements(compound).keys())

    # Create DataFrame to store stoichiometric array
    stoichiometric_df = pd.DataFrame(
        0, index=range(len(composition_column)), columns=all_elements
    )

    # Populate stoichiometric array
    for i, compound in enumerate(composition_column):
        element_counts = extract_elements(compound)
        for element, count in element_counts.items():
            stoichiometric_df.at[i, element] = count

    return stoichiometric_df


def build_mp_features(
    Y_raw: pd.DataFrame,
    PT: pd.DataFrame,
    MM: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Construct features for the Materials Project dataset.
    Matches notebook cells 45, 47, 49, and 51.
    """
    Y = Y_raw.copy()

    # Build stoichiometric array
    stoichiometric_array = build_stoichiometric_array(Y["composition"])

    # Build element-weighted features via the alloys module (as in Novamag)
    Y["stoicentw"] = al.get_StoicEntw(stoichiometric_array)
    Y["Zw"] = al.get_Zw(PT, stoichiometric_array)
    Y["periodw"] = al.get_Periodw(PT, stoichiometric_array)
    Y["groupw"] = al.get_Groupw(PT, stoichiometric_array)
    Y["meltingTw"] = al.get_MeltingTw(PT, stoichiometric_array)
    Y["miedemaH"] = al.get_Miedemaw(MM, stoichiometric_array)
    Y["valencew"] = al.get_Valencew(PT, stoichiometric_array)
    Y["electronegw"] = al.get_Electronegw(PT, stoichiometric_array)
    print("Finished building MP features")

    mp_cols = [
        "stoicentw",
        "Zw",
        "periodw",
        "groupw",
        "meltingTw",
        "miedemaH",
        "valencew",
        "electronegw",
    ]

    # Remove rows with missing target, separate target from predictors
    Y.dropna(
        axis=0,
        subset=["total_magnetization_normalized_vol"] + mp_cols,
        inplace=True,
    )

    # Round the saturation magnetization to 1.d.p
    Y["total_magnetization_normalized_vol"] = pd.to_numeric(
        Y["total_magnetization_normalized_vol"]
    ).round(decimals=2)

    # Group duplicates by chemical formula and replace values with median
    Y = Y.groupby(by="composition").median()
    Y.index = range(len(Y))

    # Define the target as the series 'mp_y' and drop this from the dataframe 'Y'
    mp_y = Y["total_magnetization_normalized_vol"]
    Y.drop(["total_magnetization_normalized_vol"], axis=1, inplace=True)

    # Final feature matrix
    mp_X = Y[mp_cols].copy()

    return mp_X, mp_y, mp_cols


def load_mp_dataset(
    csv_path: str = "magnetism_completed - magnetism_completed.csv",
    pt_path: str = "./data/Periodic-table/periodic_table.xlsx",
    mm_path: str = "./data/Miedema-model/Miedema-model-reduced.xlsx",
):
    """
    Convenience wrapper for Materials Project: load, clean, and engineer features.
    """
    PT, MM = load_periodic_tables(pt_path, mm_path)
    Y_raw = load_mp_raw(csv_path)
    X_mp, mp_y, mp_cols = build_mp_features(Y_raw, PT, MM)
    return X_mp, mp_y, mp_cols, PT, MM


# ====== Generic train/validation split ======

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.8,
    random_state: int = 0,
):
    """
    Split any dataset into train and validation sets.
    Matches notebook cells 42 and 51.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=1 - train_size,
        random_state=random_state,
    )
    return X_train, X_valid, y_train, y_valid
