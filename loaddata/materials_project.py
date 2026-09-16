"""Materials Project: one CSV export of computed magnetic properties.

`load_materials_project` converts the target to tesla and standardizes record types, preserving source metadata.
Magnetic and element-based sample selection belongs to prepdata/modeling_table.py.
"""

import numpy as np
import pandas as pd

from loaddata.tabular_access import DEFAULT_FORMULA_COLUMN, DEFAULT_TARGET_COLUMN, standardize_records

DEFAULT_CSV_PATH = "./data/materials_project/mp-data.csv"

# Materials Project reports total magnetization per unit cell volume in μB/Å^3; the target is μ_0 M in tesla.
BOHR_MAGNETON = 9.274e-24  # A·m^2
CUBIC_ANGSTROM = 1e-30  # m^3
VACUUM_PERMEABILITY = 4 * np.pi * 1e-7  # T·m/A
TESLA_PER_BOHR_MAGNETON_PER_CUBIC_ANGSTROM = (BOHR_MAGNETON / CUBIC_ANGSTROM) * VACUUM_PERMEABILITY


def read_mp_records(csv_path: str = DEFAULT_CSV_PATH, formula_column: str = DEFAULT_FORMULA_COLUMN) -> pd.DataFrame:
    """Read the Materials Project export, renaming its composition column to the shared formula column."""
    return pd.read_csv(csv_path).rename(columns={"composition": formula_column})


def load_materials_project(
    csv_path: str = DEFAULT_CSV_PATH,
    target_column: str = DEFAULT_TARGET_COLUMN,
    formula_column: str = DEFAULT_FORMULA_COLUMN,
) -> pd.DataFrame:
    """Load all Materials Project records with a shared formula column and target in tesla.

    Args:
        csv_path: Path to the Materials Project CSV export.
        target_column: Name to give the converted saturation magnetization.
        formula_column: Chemical-formula column.

    Returns:
        A frame with `formula_column`, numeric `target_column`, nullable boolean `is_magnetic`, and source metadata.
        Non-magnetic and excluded-element records remain available for configurable selection in prepdata/.
    """
    data = read_mp_records(csv_path, formula_column=formula_column)

    volumetric_moment = pd.to_numeric(data["total_magnetization_normalized_vol"], errors="coerce")
    data[target_column] = volumetric_moment * TESLA_PER_BOHR_MAGNETON_PER_CUBIC_ANGSTROM

    data["source"] = "materials_project"
    if "material_id" in data.columns:
        data["sample_id"] = data["material_id"]
    return standardize_records(data, target_column, formula_column)
