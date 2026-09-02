"""Raw dataset loaders with basic cleaning, for Novamag and Materials Project data."""

import numpy as np
import pandas as pd

from loaddata.alloy_access import import_novamag, import_periodic_table, import_miedema_weight
from prepdata.alloy_transform import formula_contains_elements


def load_elemental_data(
    pt_path: str = "./data/Periodic-table/periodic_table.xlsx",
    mm_path: str = "./data/Miedema-model/Miedema-model-reduced.xlsx",
):
    """Load the periodic table and the Miedema model."""
    periodic_table = import_periodic_table(pt_path)
    miedema_weight = import_miedema_weight(mm_path)
    return periodic_table, miedema_weight


def load_novamag_raw_data(novamag_dir):
    """Load raw Novamag JSON data and normalize missing-value markers."""
    data = import_novamag(novamag_dir)

    # Normalize literal 'none' entries
    data = data.replace({None: np.nan, "none": np.nan, "None": np.nan})
    data = data.infer_objects(copy=False)

    return data


def load_mp_raw_data(csv_path):
    """Read the Materials Project CSV, convert magnetization units, and filter invalid entries."""
    data = pd.read_csv(csv_path)
    data = data.rename(columns={"composition": "chemical formula"})
    print(f"The total number of imported features is {len(data.columns)}")

    # Convert target to saturation magnetization, μB/Å^3 → A/m → μ_0 M（T）
    print("Converting total magnetization to saturation magnetization...")
    data["total_magnetization_normalized_vol"] = pd.to_numeric(data["total_magnetization_normalized_vol"], errors='coerce')
    mu_B = 9.274e-24  # A·m^2
    angstrom3_to_m3 = 1e-30  # m^3
    mu_0 = 4 * np.pi * 1e-7  # T·m/A
    factor = (mu_B / angstrom3_to_m3) * mu_0  # ~= 11.65 T per (μB/Å^3)
    data["saturation magnetization"] = data["total_magnetization_normalized_vol"] * factor

    # Keep only magnetic systems
    initial_rows = len(data)
    data = data[data["is_magnetic"]]
    print(f"Filtered to magnetic systems: {len(data)} of {initial_rows} rows remaining")

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
    before_rare_earth = len(data)
    data = data[~formula_contains_elements(data, rare_earth_elements)]
    print(f"Removed rare-earth entries: {before_rare_earth - len(data)} rows dropped")

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
    before_non_commercial = len(data)
    data = data[~formula_contains_elements(data, non_commercial_elements)]
    print(f"Removed non-commercial entries: {before_non_commercial - len(data)} rows dropped")

    data = data[["chemical formula", "saturation magnetization"]]
    print(f"Final MP raw dataset size: {data.shape[0]} material samples x {data.shape[1]} features")
    return data
