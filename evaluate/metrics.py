"""Regression metric primitives, and the long-form table every evaluation mode reports through.

One row per (split, model, metric) keeps cross-validation, random splits and the out-of-distribution families in the
same shape, so the mean/std, the significance tests and the figures are all derived from one table rather than
re-implemented per mode.
Formatting lives in reporting.py so this module stays free of display concerns.
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from loaddata.splits import Split
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

# The paired tests read a positive difference as "model_a is worse", so they only apply to metrics where lower wins.
# Derived rather than listed, so a metric added to METRICS is compared unless it is explicitly higher-is-better.
HIGHER_IS_BETTER = ("r2",)
COMPARABLE_METRICS = tuple(metric for metric in METRICS if metric not in HIGHER_IS_BETTER)


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
# Per-model, per-metric scores: {model name: {metric: [one score per split]}}.
FoldScores = Dict[str, Dict[str, List[float]]]

RESULT_COLUMNS = ("scenario", "split_type", "split_id", "seed", "model", "metric", "value")

# What a summary is grouped by unless the caller says otherwise: one row per model and metric within a family.
SUMMARY_GROUPS = ("scenario", "split_type", "model", "metric")


def wide_table_to_results(
    table: pd.DataFrame,
    value_columns: Dict[str, str],
    *,
    identifiers: Dict[str, str],
) -> pd.DataFrame:
    """Reshape a mode's wide result table into the long form, without recomputing anything.

    The values are carried over as they are, so a table already written to disk and its long-form twin cannot drift.

    Args:
        table: One row per split, with a column per metric.
        value_columns: {column in `table`: metric name in the output}.
        identifiers: {column in RESULT_COLUMNS: column in `table`, or a constant if absent from it}.

    Returns:
        A frame with exactly RESULT_COLUMNS, empty if `table` is.
    """
    if table.empty:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))

    frame = pd.DataFrame(index=table.index)
    for target, source in identifiers.items():
        frame[target] = table[source] if source in table.columns else source

    melted = []
    for column, metric in value_columns.items():
        if column not in table.columns:
            continue
        part = frame.copy()
        part["metric"] = metric
        part["value"] = pd.to_numeric(table[column], errors="coerce")
        melted.append(part)

    if not melted:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))
    return pd.concat(melted, ignore_index=True)[list(RESULT_COLUMNS)]


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


def results_to_scores(results: pd.DataFrame) -> Dict[str, Dict[str, List[float]]]:
    """Re-express long-form rows as {model: {metric: [one value per split]}} — the inverse of build_result_rows.

    Row order is preserved, and that is what makes two models' lists paired: the i-th value of each comes from the same
    split. Pass rows that line up — one scenario, and one seed where the splits are per-seed — or the pairing is
    meaningless. This is the shape the paired significance tests take, so a comparison can be recomputed from a saved
    result table without re-running anything.

    Args:
        results: Rows shaped like RESULT_COLUMNS.

    Returns:
        {model name: {metric: [value per row, in table order]}}.
    """
    return {
        model: {
            metric: [float(value) for value in per_metric["value"]]
            for metric, per_metric in per_model.groupby("metric", sort=False)
        }
        for model, per_model in results.groupby("model", sort=False)
    }


def best_model(summary: pd.DataFrame, metric: str = "mse") -> str:
    """Return the model name with the best mean of `metric` in a summary table.

    Raises:
        ValueError: If the summary has no rows for `metric`.
    """
    rows = summary[summary["metric"] == metric]
    if rows.empty:
        raise ValueError(f"No {metric!r} rows in the summary; metrics present: {sorted(summary['metric'].unique())}")
    ranked = rows.sort_values("mean", ascending=metric not in HIGHER_IS_BETTER)
    return str(ranked.iloc[0]["model"])


def scores_to_results(
    scores: FoldScores,
    splits: Sequence[Split],
    *,
    scenario: str,
    split_type: str = "ID",
    seed: int = 0,
) -> pd.DataFrame:
    """Re-express per-split scores as the long-form table every mode reports through.

    Args:
        scores: What cross_validate_models returned for `splits`.
        splits: The same splits, read for their identifiers.
        scenario: Split family name carried into the table.
        split_type: Shifted or in-distribution; plain K-fold and random splits are in-distribution.
        seed: The seed that produced `splits`.

    Returns:
        A frame shaped like evaluate.metrics.RESULT_COLUMNS.
    """
    return build_result_rows(
        scores,
        scenario=scenario,
        split_type=split_type,
        split_ids=[split_id for split_id, _, _ in splits],
        seed=seed,
    )


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
