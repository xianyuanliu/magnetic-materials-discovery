"""Engineer alloy features shared across Novamag and Materials Project data.

Featurization (`add_engineered_features`) is separated from the target-side
cleaning (`build_features`) because inference has compositions but no target:
`pipeline/predict_pipeline.py` and `interpret/case_studies.py` need the same
nine features for formulas that have never been measured.
"""

from typing import List, Tuple

import pandas as pd

from prepdata import alloy_transform

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

# Alloys below this saturation magnetization are treated as non-magnetic and
# dropped (cutoff determined through prior model optimization).
DEFAULT_MIN_TARGET = 0.18


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
        mm: Symmetrised Miedema mixing-enthalpy matrix.
        formula_column: Name of the chemical-formula column.

    Returns:
        A copy of `raw_data` with the ENGINEERED_FEATURE_COLUMNS added. A row
        whose formula cannot be parsed gets NaN features rather than a
        plausible-looking zero vector; see prepdata/alloy_transform.py.

    Raises:
        ValueError: If `formula_column` is missing.
    """
    if formula_column not in raw_data.columns:
        raise ValueError(f"Missing column {formula_column!r} in the input data.")

    data = raw_data.copy()
    stoich_array = alloy_transform.get_stoich_array(data, pt, formula_column=formula_column)

    data["compoundradix"] = alloy_transform.get_compound_radix(data, formula_column=formula_column)
    data["stoicentw"] = alloy_transform.get_stoic_entw(stoich_array)         # mixing entropy
    data["Zw"] = alloy_transform.get_zw(pt, stoich_array)                    # atomic weight
    data["periodw"] = alloy_transform.get_periodw(pt, stoich_array)          # period
    data["groupw"] = alloy_transform.get_groupw(pt, stoich_array)            # group
    data["meltingTw"] = alloy_transform.get_melting_tw(pt, stoich_array)     # melting point
    data["miedemaH"] = alloy_transform.get_miedemaw(mm, stoich_array)        # Miedema enthalpy
    data["valencew"] = alloy_transform.get_valencew(pt, stoich_array)        # valence
    data["electronegw"] = alloy_transform.get_electronegw(pt, stoich_array)  # electronegativity

    return data


def build_features(
    raw_data: pd.DataFrame,
    pt: pd.DataFrame,
    mm: pd.DataFrame,
    min_target: float = DEFAULT_MIN_TARGET,
    target_column: str = "saturation magnetization",
    formula_column: str = "chemical formula",
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the modelling table: engineered features plus target-side cleaning.

    Args:
        raw_data: Frame carrying `formula_column` and `target_column`.
        pt: Periodic table data, indexed by element symbol.
        mm: Symmetrised Miedema mixing-enthalpy matrix.
        min_target: Rows below this target value are dropped as non-magnetic.
        target_column: Name of the measured property being modelled.
        formula_column: Name of the chemical-formula column.

    Returns:
        (data, feature_columns), where `data` is indexed by formula with one row
        per distinct composition (duplicates collapsed by median).
    """
    data = add_engineered_features(raw_data, pt, mm, formula_column=formula_column)

    feature_columns = list(ENGINEERED_FEATURE_COLUMNS)
    data = data.dropna(axis=0, subset=[target_column] + feature_columns)
    data = data[data[target_column] >= min_target]

    # Collapse duplicate formulas by taking the median feature values.
    data = data.groupby(by=formula_column).median()

    return data, feature_columns
