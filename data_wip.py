# data.py
"""
Data loading and feature engineering helpers for Novamag and Materials Project datasets:
- Raw loaders with basic cleaning
- Shared alloy feature engineering
- Generic stoichiometric array builder
- Train/validation split helper
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """Load dataset from the given path."""
    data = pd.read_csv(path)

    data.index = len(data)
    ground_truth = data["saturation magnetization"]
    data.drop(["saturation magnetization"], axis=1, inplace=True)

    # Keep only the feature columns while preserving the column order
    data = data[feature_columns].copy()
    return data, ground_truth


# ====== Shared train/validation split ======

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.8,
    random_state: int = 0,
):
    """
    Split any dataset into train and validation sets.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=1 - train_size,
        random_state=random_state,
    )
    return X_train, X_valid, y_train, y_valid


# ====== Shared stoichiometric array builder ======
pattern = r"([A-Z][a-z]*)(\d*)"

def extract_elements(compound: str):
    """Extract elements and their counts from a formula string."""
    element_counts = {}
    matches = re.findall(pattern, compound)
    for match in matches:
        element = match[0]
        count = int(match[1]) if match[1] else 1
        element_counts[element] = count
    return element_counts


def build_stoichiometric_array(composition_column: pd.Series) -> pd.DataFrame:
    """
    Build the stoichiometric array for a composition column.
    """
    # Extract unique element symbols from all compositions
    all_elements = set()
    for compound in composition_column:
        all_elements.update(extract_elements(compound).keys())

    # Create DataFrame to store stoichiometric array
    stoichiometric_df = pd.DataFrame(0, index=range(len(composition_column)), columns=all_elements)

    # Populate stoichiometric array
    for i, compound in enumerate(composition_column):
        element_counts = extract_elements(compound)
        for element, count in element_counts.items():
            stoichiometric_df.at[i, element] = count

    return stoichiometric_df