"""Novamag: a directory of JSON records, one per computed structure.

`read_novamag_records` keeps every field for inspection; `load_novamag` narrows it to the shape described in
loaddata/unified.py, which is what prepdata/ consumes.

Adapted from https://github.com/rich970/ML-alloy-design/blob/master/alloys.py with modifications.
"""

import os
import warnings

import numpy as np
import pandas as pd

from loaddata.unified import DEFAULT_FORMULA_COLUMN, DEFAULT_TARGET_COLUMN, select_unified_columns

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
    for dir_name, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if not file_name.endswith(".json"):
                continue
            path = os.path.join(dir_name, file_name)
            try:
                record = pd.read_json(path, encoding="Latin")
                rows.append({
                    **record.properties.chemistry,
                    **record.properties.crystal,
                    **record.properties.magnetics,
                })
            except ValueError:
                failed.append(path)

    if failed:
        warnings.warn(f"Skipped {len(failed)} unreadable Novamag file(s), the first being {failed[0]}.")

    data = pd.DataFrame(rows).apply(lambda column: column.map(_flatten))
    data = data.replace({None: np.nan, "none": np.nan, "None": np.nan})
    return data.infer_objects(copy=False)


def load_novamag(
    root_dir: str = DEFAULT_ROOT_DIR,
    target_column: str = DEFAULT_TARGET_COLUMN,
    formula_column: str = DEFAULT_FORMULA_COLUMN,
) -> pd.DataFrame:
    """Load Novamag into the unified (formula, target) frame that prepdata/ expects.

    Args:
        root_dir: Directory tree to walk for .json files.
        target_column: Measured property to carry through.
        formula_column: Chemical-formula column.

    Returns:
        A frame with exactly `formula_column` and `target_column`.

    Raises:
        ValueError: If either column is absent from the records.
    """
    return select_unified_columns(read_novamag_records(root_dir), target_column, formula_column)
