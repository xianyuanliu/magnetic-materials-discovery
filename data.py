"""Data loading and feature engineering helpers for alloy datasets."""

import re
import warnings
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pymatgen.core.composition import Composition

# ====== Formula parsing and periodic table mapping ======


def parse_elements_from_formula(formula: str) -> List[str]:
    """Extract unique element symbols from a chemical formula using pymatgen.

    Uses the same parser as alloys.get_stoich_array so element identification
    is consistent across the pipeline.

    A row whose elements come back empty is invisible to any element/period/
    group-based OOD split (it can never be selected as train or test for a
    given target), so a genuine parse failure is surfaced via warnings.warn
    rather than swallowed. A missing formula (None/NaN) is not a failure and
    stays silent.

    Example:
        Nd2Fe14B -> ["Nd", "Fe", "B"]
    """
    if pd.isna(formula):
        return []
    try:
        comp = Composition(str(formula))
        return [str(el) for el in comp.elements]
    except Exception as exc:
        warnings.warn(
            f"Could not parse chemical formula {formula!r}; treating it as "
            f"containing no elements, so it will be excluded from any "
            f"element/period/group-based OOD split ({exc})."
        )
        return []


def extract_elements_series(
    df_raw: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> List[List[str]]:
    """
    Return element list per sample aligned with dataframe rows.
    """
    if formula_column not in df_raw.columns:
        raise ValueError(f"Missing column: {formula_column}")

    formulas = df_raw[formula_column].tolist()
    elements_per_row = [parse_elements_from_formula(x) for x in formulas]

    n_unparsed = sum(
        1
        for formula, elements in zip(formulas, elements_per_row)
        if not elements and not pd.isna(formula)
    )
    if n_unparsed:
        print(
            f"[WARN] {n_unparsed} of {len(formulas)} formulas in "
            f"'{formula_column}' could not be parsed and will be excluded "
            f"from any element/period/group-based filtering or splitting."
        )

    return elements_per_row


def elements_mask(
    elements_per_row: Sequence[Sequence[str]],
    elements: Iterable[str],
) -> np.ndarray:
    """Boolean mask: True where a row's parsed elements intersect `elements`.

    Built on the same pymatgen-based parsing as parse_elements_from_formula,
    so element-based filtering/grouping stays consistent with the OOD element
    splits instead of relying on ad hoc formula substring matching.
    """
    target = set(elements)
    return np.array(
        [bool(target.intersection(els)) for els in elements_per_row],
        dtype=bool,
    )


def formula_contains_elements(
    df: pd.DataFrame,
    elements: Iterable[str],
    formula_column: str = "chemical formula",
) -> np.ndarray:
    """Boolean mask over `df` rows whose formula contains any of `elements`."""
    elements_per_row = extract_elements_series(df, formula_column=formula_column)
    return elements_mask(elements_per_row, elements)


def load_periodic_table_map(
    pt_path: str,
    element_col: str = "symbol",
    period_col: str = "period",
    group_block_col: str = "group_block",
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Load periodic table mapping from periodic_table.xlsx.

    Your file contains:
        symbol
        period
        group_block (string like "group 1, s-block")

    Returns:
        element_to_group
        element_to_period
    """
    pt = pd.read_excel(pt_path)

    for c in [element_col, period_col, group_block_col]:
        if c not in pt.columns:
            raise ValueError(
                f"Missing column '{c}' in periodic table file. "
                f"Found: {list(pt.columns)}"
            )

    pt = pt[[element_col, period_col, group_block_col]].copy()

    pt[element_col] = pt[element_col].astype(str)
    pt[period_col] = pt[period_col].astype(int)

    def _extract_group(gb):
        if pd.isna(gb):
            return None

        m = re.search(r"group\s*(\d+)", str(gb), flags=re.IGNORECASE)

        return int(m.group(1)) if m else None

    pt["group"] = pt[group_block_col].map(_extract_group)

    element_to_period = dict(zip(pt[element_col], pt[period_col]))

    element_to_group = {
        sym: int(g)
        for sym, g in zip(pt[element_col], pt["group"])
        if pd.notnull(g)
    }

    return element_to_group, element_to_period


# def load_features_target_and_formulas(
#     path: str,
#     target_column: str = "saturation magnetization",
#     formula_column: str = "chemical formula",
# ):
#     """Load X, y and formulas aligned by index."""
#     df = pd.read_csv(path).reset_index(drop=True)
#     if target_column not in df.columns:
#         raise ValueError(f"Missing target column '{target_column}'")
#     if formula_column not in df.columns:
#         raise ValueError(f"Missing formula column '{formula_column}'")
#     y = df[target_column]
#     formulas = df[formula_column].astype(str)
#     feature_columns = df.columns.drop([target_column, formula_column])
#     X = df[feature_columns].copy()
#     return X, y, formulas, feature_columns

# ====== Data loading ======
def load_features_and_target(path, target_column="saturation magnetization"):
    """Load features, target and drop formula column."""
    data = pd.read_csv(path).reset_index(drop=True)

    ground_truth = data[target_column]
    feature_columns = data.columns.drop([target_column, "chemical formula"])

    data = data[feature_columns].copy()
    return data, ground_truth, feature_columns


def load_train_test_features_and_target(
    train_path,
    test_path,
    target_column="saturation magnetization",
):
    """Load train/test CSVs and align feature columns."""
    train_data = pd.read_csv(train_path).reset_index(drop=True)
    test_data = pd.read_csv(test_path).reset_index(drop=True)

    train_y = train_data[target_column]
    test_y = test_data[target_column]

    feature_columns = train_data.columns.drop([target_column, "chemical formula"])
    missing_features = set(feature_columns) - set(test_data.columns)
    if missing_features:
        missing_str = ", ".join(sorted(missing_features))
        raise ValueError(f"Test data missing features: {missing_str}")

    X_train = train_data[feature_columns].copy()
    X_test = test_data[feature_columns].copy()
    return X_train, train_y, X_test, test_y, feature_columns


def load_raw_data(path):
    """Load raw dataset from a CSV."""
    data = pd.read_csv(path).reset_index(drop=True)
    return data

# ====== Train/validation split ======
def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.8,
    random_state: int = 0,
):
    """Split into train and validation sets."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=1 - train_size,
        random_state=random_state,
    )
    return X_train, X_valid, y_train, y_valid
