"""Novamag: a directory of JSON records, one per computed structure.

`read_novamag_records` flattens the source fields; `load_novamag` standardizes column names and types for prepdata/.
Source metadata is preserved, and task-specific sample selection happens in prepdata/.

Adapted from https://github.com/rich970/ML-alloy-design/blob/master/alloys.py with modifications.
"""

import os
import warnings

import numpy as np
import pandas as pd

from loaddata.tabular_access import DEFAULT_FORMULA_COLUMN, DEFAULT_TARGET_COLUMN, standardize_records

DEFAULT_ROOT_DIR = "./data/novamag/Novamag_Data_Files/"


def _flatten(x):
    """Unwrap the nested {'value': ...} dicts the Novamag JSON uses for measured quantities."""
    if isinstance(x, dict) and "value" in x and len(x) == 1:
        return x["value"]
    return x


def read_novamag_records(root_dir: str = DEFAULT_ROOT_DIR) -> pd.DataFrame:
    """Read every Novamag JSON file into one frame, keeping all chemistry, crystal and magnetic fields.

    Unreadable files are counted and reported once rather than aborting the walk, because the collection carries a
    handful of malformed records that are not worth failing a whole run over.

    Args:
        root_dir: Directory tree to walk for .json files.

    Returns:
        One row per record, with the nested value-dicts flattened and missing-value markers normalized to NaN.
    """
    rows, failed = [], []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Novamag directory does not exist: {root_dir}")
    for dir_name, sub_dirs, file_names in os.walk(root_dir):
        sub_dirs.sort()
        for file_name in sorted(file_names):
            if not file_name.endswith(".json"):
                continue
            path = os.path.join(dir_name, file_name)
            try:
                record = pd.read_json(path, encoding="Latin")
                rows.append({
                    **record.properties.chemistry,
                    **record.properties.crystal,
                    **record.properties.magnetics,
                    "sample_id": os.path.relpath(path, root_dir),
                    "source": "novamag",
                })
            except ValueError:
                failed.append(path)

    if failed:
        warnings.warn(f"Skipped {len(failed)} unreadable Novamag file(s), the first being {failed[0]}.")

    if not rows:
        raise ValueError(f"No readable Novamag records found in {root_dir}")
    data = pd.DataFrame(rows).apply(lambda column: column.map(_flatten))
    data = data.replace({None: np.nan, "none": np.nan, "None": np.nan})
    return data.infer_objects(copy=False)


def load_novamag(
    root_dir: str = DEFAULT_ROOT_DIR,
    target_column: str = DEFAULT_TARGET_COLUMN,
    formula_column: str = DEFAULT_FORMULA_COLUMN,
) -> pd.DataFrame:
    """Load Novamag with shared formula and numeric target columns, preserving source metadata.

    Args:
        root_dir: Directory tree to walk for .json files.
        target_column: Output name for saturation magnetization (already in tesla in the source).
        formula_column: Output name for the chemical-formula column.

    Returns:
        Standardized records with `formula_column`, `target_column`, source identifiers and other metadata.

    Raises:
        ValueError: If either column is absent from the records.
    """
    data = read_novamag_records(root_dir).rename(columns={
        DEFAULT_FORMULA_COLUMN: formula_column,
        DEFAULT_TARGET_COLUMN: target_column,
    })
    return standardize_records(data, target_column, formula_column)
