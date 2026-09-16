"""Shared data access: record standardization, element reference tables and prepared feature CSVs.

Dataset-specific readers live in novamag.py and materials_project.py. This module handles shared tabular formats;
composition parsing, sample selection and feature calculation belong to prepdata/.

Reference-table readers adapted from https://github.com/rich970/ML-alloy-design/blob/master/alloys.py.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdtypes

DEFAULT_TARGET_COLUMN = "saturation magnetization"
DEFAULT_FORMULA_COLUMN = "chemical formula"
DEFAULT_PERIODIC_TABLE_PATH = "./data/Periodic-table/periodic_table.xlsx"
DEFAULT_MIEDEMA_PATH = "./data/Miedema-model/Miedema-model-reduced.xlsx"


def standardize_records(
    data: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    formula_column: str = DEFAULT_FORMULA_COLUMN,
) -> pd.DataFrame:
    """Validate required fields and normalize types while preserving record metadata.

    Formula strings retain their original stoichiometry; equivalent compositions are grouped in prepdata/.
    Invalid or non-finite targets become NaN. Optional is_magnetic values become nullable booleans;
    unknown labels raise rather than silently becoming False. No task-specific row filtering happens here.
    Dataset readers are responsible for source column mappings and physical unit conversions.
    """
    required = [formula_column, target_column]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) {missing} in the loaded data. Available: {sorted(data.columns)}")
    data = data[required + [column for column in data.columns if column not in required]].copy()
    data[formula_column] = data[formula_column].astype("string").str.strip().replace("", pd.NA)
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "is_magnetic" in data.columns:
        labels = data["is_magnetic"].astype("string").str.strip().str.lower()
        labels = labels.replace({"": pd.NA, "none": pd.NA, "nan": pd.NA})
        mapping = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
        invalid = labels.notna() & ~labels.isin(mapping)
        if invalid.any():
            raise ValueError(f"Unknown is_magnetic label(s): {labels[invalid].unique().tolist()}")
        data["is_magnetic"] = labels.map(mapping).astype("boolean")
    return data.reset_index(drop=True)


def load_periodic_table(path: str = DEFAULT_PERIODIC_TABLE_PATH) -> pd.DataFrame:
    """Read the periodic table spreadsheet, indexed by element symbol."""
    pt = pd.read_excel(path)
    pt.index = pt["symbol"]
    return pt


def load_miedema_weight(path: str = DEFAULT_MIEDEMA_PATH) -> pd.DataFrame:
    """Read the Miedema mixing-enthalpy spreadsheet and symmetrize it.

    The sheet stores only one triangle of the element-pair matrix, so it is added to its own transpose to make
    `mm.loc[a, b]` and `mm.loc[b, a]` both resolve.
    """
    mm = pd.read_excel(path, header=1, index_col=73, usecols=range(0, 74), nrows=73).fillna(0)
    return mm + mm.transpose().fillna(0)


def load_element_properties(
    pt_path: str = DEFAULT_PERIODIC_TABLE_PATH,
    mm_path: str = DEFAULT_MIEDEMA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both reference tables at once, since the descriptors always need the pair.

    Returns:
        (periodic_table, miedema_weight).
    """
    return load_periodic_table(pt_path), load_miedema_weight(mm_path)


def resolve_feature_columns(
    data: pd.DataFrame,
    target_column: str,
    formula_column: str = "chemical formula",
    feature_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    """Determine and validate which columns are model inputs.

    Naming the features explicitly is strongly preferred: the fallback — "every column that is not the target or the
    formula" — would otherwise promote a stray id or text column into a model input.

    Args:
        data: The loaded table.
        target_column: Name of the column being predicted.
        formula_column: Name of the chemical-formula column, if present.
        feature_columns: Explicit feature list from the run config, or None to infer them.

    Returns:
        The feature column names, in the order the models will see them.

    Raises:
        ValueError: If a named feature is missing, if nothing is left after excluding the target and formula, or if any
            feature column is non-numeric.
    """
    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in data.columns]
        if missing:
            raise ValueError(
                f"feature_columns names {missing} which are not in the data. "
                f"Available: {sorted(data.columns)}"
            )
        resolved = list(feature_columns)
    else:
        excluded = {target_column, formula_column}
        resolved = [c for c in data.columns if c not in excluded]
        if not resolved:
            raise ValueError(f"No feature columns left after excluding {sorted(excluded)}.")

    non_numeric = [c for c in resolved if not pdtypes.is_numeric_dtype(data[c])]
    if non_numeric:
        raise ValueError(
            f"Non-numeric feature column(s) {non_numeric} (dtypes: "
            f"{[str(data[c].dtype) for c in non_numeric]}). Set 'feature_columns' "
            f"in the run config to name the model inputs explicitly."
        )

    return resolved


def load_features_and_target(
    path: str,
    target_column: str = "saturation magnetization",
    feature_columns: Optional[Sequence[str]] = None,
    formula_column: str = "chemical formula",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load a featurized CSV into (features, target, feature_columns).

    Args:
        path: Path to the featurized CSV.
        target_column: Name of the column being predicted.
        feature_columns: Explicit feature list, or None to infer them.
        formula_column: Name of the chemical-formula column.

    Returns:
        (X, y, feature_columns).

    Raises:
        ValueError: If the target column is missing, or feature resolution fails; see resolve_feature_columns.
    """
    data = pd.read_csv(path).reset_index(drop=True)

    if target_column not in data.columns:
        raise ValueError(f"Missing target column {target_column!r} in {path}. Available: {sorted(data.columns)}")

    resolved = resolve_feature_columns(data, target_column, formula_column, feature_columns)
    return data[resolved].copy(), data[target_column], resolved


def load_raw_data(path: str) -> pd.DataFrame:
    """Load a raw dataset from a CSV."""
    return pd.read_csv(path).reset_index(drop=True)
