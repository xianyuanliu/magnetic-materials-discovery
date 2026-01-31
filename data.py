"""Data loading and feature engineering helpers for alloy datasets."""

from os import path
import re
import pandas as pd
from sklearn.model_selection import train_test_split


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
