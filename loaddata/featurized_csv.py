"""Reader for the featurized CSV that prepdata/ saves, and the feature-column contract it is read under.

This is the re-entry point for day-to-day runs: it skips loaddata/<dataset>.py and prepdata/ by reading
the modeling table they produced, so an experiment does not re-parse the raw collection every time.
"""

from typing import List, Optional, Sequence, Tuple

import pandas as pd
from pandas.api import types as pdtypes


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
