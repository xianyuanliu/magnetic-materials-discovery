"""Shared data access: record standardization, element reference tables and prepared feature CSVs.

Dataset-specific readers live in novamag.py and materials_project.py. This module handles shared tabular formats;
composition parsing, sample selection and feature calculation belong to prepdata/.

Reference-table readers adapted from https://github.com/rich970/ML-alloy-design/blob/master/alloys.py.
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdtypes

DEFAULT_TARGET_COLUMN = "saturation magnetization"
DEFAULT_FORMULA_COLUMN = "chemical formula"

# Columns a feature table carries for traceability rather than for the model. They are numeric or id-like, so the
# fallback in resolve_feature_columns would otherwise promote them into model inputs and leak row provenance.
METADATA_COLUMNS = ("sample_id", "n_records", "source")
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

    Naming the features explicitly is strongly preferred: the fallback — "every remaining column" — would otherwise
    promote a stray id or text column into a model input. It skips the target, the formula and METADATA_COLUMNS, but
    it cannot recognize a column this project has never seen.

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
        if not list(feature_columns):
            raise ValueError("feature_columns was given but is empty, which would leave the model no inputs.")
        missing = [c for c in feature_columns if c not in data.columns]
        if missing:
            raise ValueError(
                f"feature_columns names {missing} which are not in the data. "
                f"Available: {sorted(data.columns)}"
            )
        resolved = list(feature_columns)
    else:
        excluded = {target_column, formula_column, *METADATA_COLUMNS}
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


def load_feature_table(
    paths,
    target_column: str = DEFAULT_TARGET_COLUMN,
    formula_column: str = DEFAULT_FORMULA_COLUMN,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Read one or more feature tables written by build_feature_tables.py.

    Several paths are concatenated into one pool. Which rows end up in training and which in test is decided by the
    evaluation mode, so a train/test pair of files is only a way of naming the data, not a split that is honored here.

    Args:
        paths: One path, or an iterable of paths to concatenate.
        target_column: Name of the column being predicted.
        formula_column: Name of the chemical-formula column, absent for tasks that have none.
        feature_columns: Explicit model inputs, or None to infer and validate them.

    Returns:
        (X, y, metadata). X holds exactly the resolved feature columns in the order the models will see them, so the
        feature names are `list(X.columns)`. metadata holds every remaining column — the formula, and whatever the
        table records about where each sample came from — row-aligned with X, so splits and a second modality can key
        off the same samples without re-reading the file.

    Raises:
        ValueError: If no path is given, the target column is missing, or the features do not resolve to a usable
            numeric matrix.
    """
    if paths is None or isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [path for path in paths if path]
    if not paths:
        raise ValueError("load_feature_table needs at least one path.")

    # Deduplicated because a config may name the same file as both train and test.
    unique_paths, seen = [], set()
    for path in paths:
        key = os.path.normpath(str(path))
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)

    frames = [pd.read_csv(path) for path in unique_paths]
    data = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    data = data.reset_index(drop=True)

    if target_column not in data.columns:
        raise ValueError(
            f"Missing target column {target_column!r} in {unique_paths}. Available: {sorted(data.columns)}"
        )

    resolved = resolve_feature_columns(data, target_column, formula_column, feature_columns)
    metadata = [column for column in data.columns if column not in resolved and column != target_column]
    return data[resolved].copy(), data[target_column].copy(), data[metadata].copy()
