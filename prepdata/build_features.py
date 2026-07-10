"""Engineer alloy features shared across Novamag and Materials Project data."""

import pandas as pd

import prepdata.alloy_transform as alloy_transform
from loaddata.raw_loaders import load_elemental_data


def build_features(
    raw_data: pd.DataFrame,
    pt: pd.DataFrame,
    mm: pd.DataFrame,
    min_saturation_magnetization: float = 0.18,
):
    """Engineer alloy features shared across Novamag and Materials Project data."""
    data = raw_data.copy()

    # compoundradix: number of species in the compound (binary, ternary, etc.)
    data["compoundradix"] = alloy_transform.get_compound_radix(data)

    # stoichiometric array
    stoich_array = alloy_transform.get_stoich_array(data, pt)

    # Compute element-weighted features from the stoichiometric array
    data["stoicentw"] = alloy_transform.get_stoic_entw(stoich_array)           # mixing entropy
    data["Zw"] = alloy_transform.get_zw(pt, stoich_array)                     # atomic weight
    data["periodw"] = alloy_transform.get_periodw(pt, stoich_array)           # period
    data["groupw"] = alloy_transform.get_groupw(pt, stoich_array)             # group
    data["meltingTw"] = alloy_transform.get_melting_tw(pt, stoich_array)       # melting point
    data["miedemaH"] = alloy_transform.get_miedemaw(mm, stoich_array)         # Miedema mixing enthalpy
    data["valencew"] = alloy_transform.get_valencew(pt, stoich_array)         # valence
    data["electronegw"] = alloy_transform.get_electronegw(pt, stoich_array)   # electronegativity

    feature_columns = [
        "compoundradix",
        "stoicentw",
        "Zw",
        "periodw",
        "groupw",
        "meltingTw",
        "miedemaH",
        "valencew",
        "electronegw",
    ]

    # Remove rows with missing target
    data = data.dropna(
        axis=0,
        subset=["saturation magnetization"] + feature_columns,
    )

    # Drop alloys which are below magnetic cutoff (determined through prior model optimization)
    data = data.drop(data[data["saturation magnetization"] < min_saturation_magnetization].index, axis=0)

    # Collapse duplicate chemical formulas by taking the median feature values
    data = data.groupby(by="chemical formula").median()

    return data, feature_columns


def process_data(raw_data, pt_path, mm_path):
    """Load elemental data and engineer alloy features for a raw magnetism dataset."""
    periodic_table, miedema_weight = load_elemental_data(pt_path, mm_path)
    data, _ = build_features(raw_data, periodic_table, miedema_weight)
    print(f"The total number of samples after cleaning is {len(data)}")

    return data
