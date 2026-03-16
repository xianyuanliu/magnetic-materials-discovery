"""Data loading and feature engineering helpers for alloy datasets."""

import re
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# ====== Formula parsing and periodic table mapping ======

_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")


def parse_elements_from_formula(formula: str) -> List[str]:
    """
    Extract element tokens from chemical formula.

    Example:
        Nd2Fe14B -> ["Nd", "Fe", "B"]
    """
    if formula is None:
        return []

    tokens = _ELEMENT_RE.findall(str(formula))

    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


def extract_elements_series(
    df_raw: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> List[List[str]]:
    """
    Return element list per sample aligned with dataframe rows.
    """
    if formula_column not in df_raw.columns:
        raise ValueError(f"Missing column: {formula_column}")

    return [
        parse_elements_from_formula(x)
        for x in df_raw[formula_column].tolist()
    ]


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


def load_features_target_and_formulas(
    path: str,
    target_column: str = "saturation magnetization",
    formula_column: str = "chemical formula",
):
    """
    Load X, y and formulas aligned by index.
    Needed for OOD evaluation.
    """
    df = pd.read_csv(path).reset_index(drop=True)

    if target_column not in df.columns:
        raise ValueError(f"Missing target column '{target_column}'")

    if formula_column not in df.columns:
        raise ValueError(f"Missing formula column '{formula_column}'")

    y = df[target_column]

    formulas = df[formula_column].astype(str)

    feature_columns = df.columns.drop([target_column, formula_column])

    X = df[feature_columns].copy()

    return X, y, formulas, feature_columns

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
