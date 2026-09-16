"""The frame shape every dataset module under loaddata/ returns.

Unifying the datasets is this stage's job: prepdata/ should not need to know which collection a row came from. The
contract is deliberately minimal — the chemical formula and the measured target, with both column names configurable,
so a project predicting a different property reuses the same dataset modules.

Extra columns are dropped here rather than downstream because prepdata/modeling_table.py collapses duplicate formulas
with a median, which would silently aggregate anything that survived this far.

To add a dataset, write loaddata/<name>.py exposing `load_<name>(path, target_column, formula_column)` that ends in a
`select_unified_columns` call.
"""

import pandas as pd

DEFAULT_TARGET_COLUMN = "saturation magnetization"
DEFAULT_FORMULA_COLUMN = "chemical formula"


def select_unified_columns(
    data: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    formula_column: str = DEFAULT_FORMULA_COLUMN,
) -> pd.DataFrame:
    """Narrow a raw dataset frame to the two columns every dataset module returns.

    Args:
        data: The dataset module's cleaned frame, which may carry any number of extra columns.
        target_column: Measured property to carry through.
        formula_column: Chemical-formula column.

    Returns:
        A frame with exactly `formula_column` and `target_column`, in that order.

    Raises:
        ValueError: If either column is absent.
    """
    missing = [column for column in (formula_column, target_column) if column not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) {missing} in the loaded data. Available: {sorted(data.columns)}")
    return data[[formula_column, target_column]]
