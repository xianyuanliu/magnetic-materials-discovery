"""Regression metric primitives: MSE / MAE / MRE / R², and mean±std formatting."""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format a "mean ± std" string, or "nan" if either value is missing."""
    if mean is None or std is None or np.isnan(mean) or np.isnan(std):
        return "nan"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


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
