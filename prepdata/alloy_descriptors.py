"""The nine alloy descriptors this project predicts saturation magnetization from.

Which quantities to compute is a choice made for this task; the machinery that computes them is task-agnostic and lives
in prepdata/composition.py. Another materials task should import that module and define its own descriptor set here.

Element-weighted calculations are adapted from
https://github.com/rich970/ML-alloy-design/blob/master/alloys.py with modifications.
"""

from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd

from prepdata.composition import (
    get_atomic_fraction,
    get_compound_radix,
    get_mixing_entropy,
    get_stoich_array,
    get_weighted_property,
)

# The engineered feature vector, in a fixed order. Exported so run configs can name the columns explicitly instead of
# relying on "every column that is not the target or the formula", which silently promoted any stray id or metadata
# column into a model input.
ENGINEERED_FEATURE_COLUMNS: Tuple[str, ...] = (
    "compoundradix",
    "stoicentw",
    "Zw",
    "periodw",
    "groupw",
    "meltingTw",
    "miedemaH",
    "valencew",
    "electronegw",
)


def _numeric_property(pt, column):
    """Read one element property column as floats, tolerating the prose the spreadsheet stores it in.

    Some columns are annotated rather than numeric — "Pauling scale: 2.20", "912±3 K (639±3 °C, 1182±5 °F)" — so the
    leading number is extracted and everything else, including "no data", becomes NaN. A NaN drops the composition in
    prepdata/feature_table.py instead of contributing an invented figure to a weighted mean.
    """
    values = pt[column]
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    return pd.to_numeric(values.astype(str).str.extract(r"(-?\d*\.?\d+)")[0], errors="coerce")


def get_electronegw(pt, stoich_array):
    """Calculate element-weighted electronegativity."""
    return get_weighted_property(_numeric_property(pt, "electronegativity"), stoich_array)


def get_zw(pt, stoich_array):
    """Calculate element-weighted atomic weight."""
    return get_weighted_property(pt["atomic_weight"], stoich_array)


def get_periodw(pt, stoich_array):
    """Calculate element-weighted period number."""
    return get_weighted_property(pt["period"], stoich_array)


def get_melting_tw(pt, stoich_array):
    """Calculate element-weighted melting temperature."""
    return get_weighted_property(_numeric_property(pt, "melting_point"), stoich_array)


def get_valencew(pt, stoich_array):
    """Calculate element-weighted valence electron number."""
    return get_weighted_property(pt["valence"], stoich_array)


def get_groupw(pt, stoich_array):
    """Calculate element-weighted group number.

    Not a plain get_weighted_property: elements with no group number (the f-block, as the spreadsheet spells it) are
    dropped and the remaining atomic fractions re-normalized, so a lanthanide does not drag the average down.
    """
    group_block = pt["group_block"].str.extract(r"(\d+)")[0]
    group_block = group_block.astype(float)
    groupw = pd.Series(index=stoich_array.index, dtype=float)
    for i, compound in stoich_array.iterrows():
        fraction = get_atomic_fraction(compound)

        if fraction.empty:
            groupw.loc[i] = np.nan
            continue

        valid_labels = [
            el for el in fraction.index
            if el in group_block.index and not pd.isna(group_block.loc[el])
        ]
        if not valid_labels:
            groupw.loc[i] = np.nan
            continue

        af_sub = fraction.loc[valid_labels]
        af_sub = af_sub / af_sub.sum()

        groupw.loc[i] = np.dot(af_sub, group_block.loc[valid_labels])
    return groupw


def get_miedemaw(mm, stoich_array):
    """Calculate weighted Miedema enthalpy of formation (pairwise sum over elements)."""
    miedemaw = pd.Series(index=stoich_array.index, dtype=float)
    for i, compound in stoich_array.iterrows():
        fraction = get_atomic_fraction(compound)

        if fraction.empty:
            miedemaw.loc[i] = np.nan
            continue

        H = 0
        valid = True
        for a, b in combinations(fraction.index, 2):
            try:
                H += 4 * fraction[a] * fraction[b] * mm.loc[a, b]
            except KeyError:
                valid = False
                break
        miedemaw.loc[i] = H if valid else np.nan
    return miedemaw


def add_engineered_features(
    raw_data: pd.DataFrame,
    pt: pd.DataFrame,
    mm: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> pd.DataFrame:
    """Add the engineered alloy features to a frame of chemical formulas.

    Args:
        raw_data: Frame carrying `formula_column`; other columns are preserved.
        pt: Periodic table data, indexed by element symbol.
        mm: Symmetrized Miedema mixing-enthalpy matrix.
        formula_column: Name of the chemical-formula column.

    Returns:
        A copy of `raw_data` with the ENGINEERED_FEATURE_COLUMNS added. A row
        whose formula cannot be parsed gets NaN features rather than a
        plausible-looking zero vector; see prepdata/composition.py.

    Raises:
        ValueError: If `formula_column` is missing.
    """
    if formula_column not in raw_data.columns:
        raise ValueError(f"Missing column {formula_column!r} in the input data.")

    data = raw_data.copy()
    stoich_array = get_stoich_array(data, pt, formula_column=formula_column)

    data["compoundradix"] = get_compound_radix(data, formula_column=formula_column)
    data["stoicentw"] = get_mixing_entropy(stoich_array)          # mixing entropy
    data["Zw"] = get_zw(pt, stoich_array)                     # atomic weight
    data["periodw"] = get_periodw(pt, stoich_array)           # period
    data["groupw"] = get_groupw(pt, stoich_array)             # group
    data["meltingTw"] = get_melting_tw(pt, stoich_array)      # melting point
    data["miedemaH"] = get_miedemaw(mm, stoich_array)         # Miedema enthalpy
    data["valencew"] = get_valencew(pt, stoich_array)         # valence
    data["electronegw"] = get_electronegw(pt, stoich_array)   # electronegativity

    return data
