"""Regression metric primitives: MSE / MAE / MRE / R².

Formatting lives in reporting.py so this module stays free of display concerns.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# Regression metrics carried through every result table, and the decimals each is printed with. MRE is a small
# fraction, so it needs more places.
METRICS = ("mse", "mae", "mre", "r2")
METRIC_DECIMALS = {"mse": 4, "mae": 4, "mre": 6, "r2": 4}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MSE/MAE/MRE/R² using sklearn's metric implementations.

    MRE (mean relative error) is sklearn's mean_absolute_percentage_error,
    which is already expressed as a fraction (not multiplied by 100).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mre": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
