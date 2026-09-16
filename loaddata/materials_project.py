"""Materials Project: one CSV export of computed magnetic properties.

`read_mp_records` keeps every exported column; `load_materials_project` converts the target to the same unit Novamag
reports and narrows the frame to the shape described in loaddata/unified.py.

The row filtering here is dataset cleaning, not task selection: it removes entries this export should not have offered
in the first place — non-magnetic systems, and elements that cannot be sourced. Deciding which of the remaining
measurements are worth modeling belongs to prepdata/modeling_table.py.
"""

from typing import List

import numpy as np
import pandas as pd

from loaddata.unified import DEFAULT_FORMULA_COLUMN, DEFAULT_TARGET_COLUMN, select_unified_columns
from prepdata.composition import formula_contains_elements

DEFAULT_CSV_PATH = "./data/materials_project/mp-data.csv"

# Dropped because a magnet built from them is not manufacturable at scale, whatever the computed moment says.
RARE_EARTH_ELEMENTS = (
    "La", "Sc", "Dy", "Sm", "Lu", "Er", "Y", "Pr", "Nd", "Gd", "Tm", "Pm", "Ce", "Tb", "Eu", "Ho", "Yb",
)
NON_COMMERCIAL_ELEMENTS = (
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
)

# Materials Project reports total magnetization per unit cell volume in μB/Å^3; the target is μ_0 M in tesla.
BOHR_MAGNETON = 9.274e-24  # A·m^2
CUBIC_ANGSTROM = 1e-30  # m^3
VACUUM_PERMEABILITY = 4 * np.pi * 1e-7  # T·m/A
TESLA_PER_BOHR_MAGNETON_PER_CUBIC_ANGSTROM = (BOHR_MAGNETON / CUBIC_ANGSTROM) * VACUUM_PERMEABILITY


def read_mp_records(csv_path: str = DEFAULT_CSV_PATH, formula_column: str = DEFAULT_FORMULA_COLUMN) -> pd.DataFrame:
    """Read the Materials Project export, renaming its composition column to the shared formula column."""
    return pd.read_csv(csv_path).rename(columns={"composition": formula_column})


def _drop_rows_containing(data: pd.DataFrame, elements: List[str], formula_column: str) -> pd.DataFrame:
    """Drop every row whose formula contains any of `elements`."""
    return data[~formula_contains_elements(data, elements, formula_column=formula_column)]


def load_materials_project(
    csv_path: str = DEFAULT_CSV_PATH,
    target_column: str = DEFAULT_TARGET_COLUMN,
    formula_column: str = DEFAULT_FORMULA_COLUMN,
) -> pd.DataFrame:
    """Load Materials Project into the unified (formula, target) frame that prepdata/ expects.

    Args:
        csv_path: Path to the Materials Project CSV export.
        target_column: Name to give the converted saturation magnetization.
        formula_column: Chemical-formula column.

    Returns:
        A frame with exactly `formula_column` and `target_column`, covering magnetic systems built from sourceable
        elements.
    """
    data = read_mp_records(csv_path, formula_column=formula_column)

    volumetric_moment = pd.to_numeric(data["total_magnetization_normalized_vol"], errors="coerce")
    data[target_column] = volumetric_moment * TESLA_PER_BOHR_MAGNETON_PER_CUBIC_ANGSTROM

    data = data[data["is_magnetic"]]
    data = _drop_rows_containing(data, list(RARE_EARTH_ELEMENTS), formula_column)
    data = _drop_rows_containing(data, list(NON_COMMERCIAL_ELEMENTS), formula_column)

    return select_unified_columns(data, target_column, formula_column)
