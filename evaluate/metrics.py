"""Regression metric primitives, and the long-form table every evaluation mode reports through.

One row per (split, model, metric) keeps cross-validation, holdout and the OOD families in the same shape, so the
mean/std, the significance tests and the figures are all derived from one table rather than re-implemented per mode.
Formatting lives in reporting.py so this module stays free of display concerns.
"""

from typing import Dict, Sequence

import numpy as np
import pandas as pd
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


# One observation per row: the identifying columns, then the metric and its value. `scenario` names the split family
# ("cross_validation", "LOEO"), `split_type` separates a shifted test set from its in-distribution references, and
# `split_id` identifies the individual split within the family.
RESULT_COLUMNS = ("scenario", "split_type", "split_id", "seed", "model", "metric", "value")

# What a summary is grouped by unless the caller says otherwise: one row per model and metric within a family.
SUMMARY_GROUPS = ("scenario", "split_type", "model", "metric")


def build_result_rows(
    scores: Dict[str, Dict[str, Sequence[float]]],
    *,
    scenario: str,
    split_type: str,
    split_ids: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Turn per-split scores into long-form rows.

    Args:
        scores: {model name: {metric: [one value per split]}}, in `split_ids` order.
        scenario: Split family this batch belongs to.
        split_type: Shifted or in-distribution, as reported in the output tables.
        split_ids: Identifier per split, same length as each metric's list.
        seed: The seed that produced these splits.

    Returns:
        A frame with exactly RESULT_COLUMNS.

    Raises:
        ValueError: If a metric's values and `split_ids` differ in length.
    """
    rows = []
    for model, per_metric in scores.items():
        for metric, values in per_metric.items():
            if len(values) != len(split_ids):
                raise ValueError(
                    f"{model}/{metric} has {len(values)} value(s) but {len(split_ids)} split id(s) were given."
                )
            for split_id, value in zip(split_ids, values):
                rows.append((scenario, split_type, split_id, seed, model, metric, float(value)))
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


def summarize_scores(results: pd.DataFrame, by: Sequence[str] = SUMMARY_GROUPS) -> pd.DataFrame:
    """Average a long-form result table into one row per group.

    Args:
        results: Rows shaped like RESULT_COLUMNS.
        by: Identifying columns to group on.

    Returns:
        `by` plus mean, std and n. std is 0.0 for a single observation, where the sample std is undefined.
    """
    if results.empty:
        return pd.DataFrame(columns=[*by, "mean", "std", "n"])

    grouped = results.groupby(list(by), sort=True)["value"]
    summary = grouped.agg(mean="mean", std=lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n="size")
    return summary.reset_index()
