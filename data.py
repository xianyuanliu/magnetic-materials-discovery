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

import alloys
import mp_alloys  # only used in the MP case study


# ====== Shared utilities: load element tables ======

def load_elemental_data(
    pt_path: str = "./data/Periodic-table/periodic_table.xlsx",
    mm_path: str = "./data/Miedema-model/Miedema-model-reduced.xlsx",
):
    """Load the periodic table and the Miedema model."""
    periodic_table = alloys.import_periodic_table(pt_path)
    miedema_weight = alloys.import_miedema_weight(mm_path)
    return periodic_table, miedema_weight


# ====== Novamag section ======

def load_novamag_raw(novamag_dir: str) -> pd.DataFrame:
    """
    Import the raw Novamag data, keeping only chemical formula and saturation magnetisation.
    Matches notebook cells 7, 8, 9, and 10.
    """
    data = alloys.importNovamag(novamag_dir)

    print(f"The total number of imported features is {len(data.columns)}")

    # Fix the 'none' values without triggering fillna downcast warnings
    data = data.replace({None: np.nan, "none": np.nan, "None": np.nan})
    data = data.infer_objects(copy=False)

    # Find columns with missing values
    na_cols = [col for col in data.columns if data[col].isna().any()]
    print(
        "The number of features with at least one NaN value is "
        f"{len(na_cols)}"
    )

    # Drop columns with more than 10 NaN values
    dropped_cols = []
    for col in na_cols:
        count = data[col].isna().sum()
        print(f"Column '{col}' has {count} nan values")
        if count > 10:
            print(f"Dropping column '{col}'")
            dropped_cols.append(col)
            data = data.drop(col, axis=1).copy()
    print(f"Number of dropped features is: {len(dropped_cols)}")

    # Drop all columns except target and chemical formula
    data = data[["chemical formula", "saturation magnetization"]]
    return data


def build_novamag_features(
    raw_data: pd.DataFrame,
    pt: pd.DataFrame,
    mm: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Engineer features for Novamag and return X_feat, y, and the feature list.
    Matches notebook cells 18, 36, 38, and 40.
    """
    data = raw_data.copy()

    # compoundradix: number of species in the compound (binary, ternary, etc.)
    data["compoundradix"] = alloys.get_CompoundRadix(pt, data)

    # stoichiometric array
    stoich_array = alloys.get_stoich_array(data, pt)

    # Compute element-weighted features from the stoichiometric array
    data["stoicentw"] = alloys.get_StoicEntw(stoich_array)           # mixing entropy
    data["Zw"] = alloys.get_Zw(pt, stoich_array)                     # atomic weight
    # data['compoundradix'] was computed above
    data["periodw"] = alloys.get_Periodw(pt, stoich_array)           # period
    data["groupw"] = alloys.get_Groupw(pt, stoich_array)             # group
    data["meltingTw"] = alloys.get_MeltingTw(pt, stoich_array)       # melting point
    data["miedemaH"] = alloys.get_Miedemaw(mm, stoich_array)         # Miedema mixing enthalpy
    data["valencew"] = alloys.get_Valencew(pt, stoich_array)         # valence
    data["electronegw"] = alloys.get_Electronegw(pt, stoich_array)   # electronegativity

    # Feature list kept consistent with the notebook
    novamag_feature_columns = [
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
    data.dropna(
        axis=0,
        subset=["saturation magnetization"] + novamag_feature_columns,
        inplace=True,
    )

    # Drop alloys which are below magnetic cutoff (determined through prior model optimisation)
    data.drop(data[data["saturation magnetization"] < 0.18].index, axis=0, inplace=True)

    # Round the saturation magnetization to 1.d.p
    data["saturation magnetization"] = pd.to_numeric(
        data["saturation magnetization"]
    ).round(decimals=2)

    # Group duplicates by chemical formula and replace values with median
    data = data.groupby(by="chemical formula").median()
    data.index = range(len(data))

    # Define target and drop from data
    ground_truth = data["saturation magnetization"]
    data.drop(["saturation magnetization"], axis=1, inplace=True)
    # Keep only the feature columns while preserving the column order
    data = data[novamag_feature_columns].copy()
    
    return data, ground_truth, novamag_feature_columns


def load_novamag_dataset(
    raw_data,
    pt_path: str,
    mm_path: str,
):
    """
    Convenience wrapper for Novamag: load the data and engineer the features.
    """
    pt, mm = load_elemental_data(pt_path, mm_path)
    data, ground_truth, novamag_feature_columns = build_novamag_features(raw_data, pt, mm)
    return data, ground_truth, novamag_feature_columns, pt, mm


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
    Y["stoicentw"] = alloys.get_StoicEntw(stoichiometric_array)
    Y["Zw"] = alloys.get_Zw(PT, stoichiometric_array)
    Y["periodw"] = alloys.get_Periodw(PT, stoichiometric_array)
    Y["groupw"] = alloys.get_Groupw(PT, stoichiometric_array)
    Y["meltingTw"] = alloys.get_MeltingTw(PT, stoichiometric_array)
    Y["miedemaH"] = alloys.get_Miedemaw(MM, stoichiometric_array)
    Y["valencew"] = alloys.get_Valencew(PT, stoichiometric_array)
    Y["electronegw"] = alloys.get_Electronegw(PT, stoichiometric_array)
    print("Finished building MP features")

    mp_feature_columns = [
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
        subset=["total_magnetization_normalized_vol"] + mp_feature_columns,
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
    mp_X = Y[mp_feature_columns].copy()

    return mp_X, mp_y, mp_feature_columns


def load_mp_dataset(
    csv_path: str = "magnetism_completed - magnetism_completed.csv",
    pt_path: str = "./data/Periodic-table/periodic_table.xlsx",
    mm_path: str = "./data/Miedema-model/Miedema-model-reduced.xlsx",
):
    """
    Convenience wrapper for Materials Project: load, clean, and engineer features.
    """
    periodic_table, miedema_weight = load_elemental_data(pt_path, mm_path)
    Y_raw = load_mp_raw(csv_path)
    X_mp, mp_y, mp_feature_columns = build_mp_features(Y_raw, periodic_table, miedema_weight)
    return X_mp, mp_y, mp_feature_columns, periodic_table, miedema_weight


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
