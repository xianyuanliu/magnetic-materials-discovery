"""Console formatting for regression metrics and their summaries."""

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from kalematerials.evaluate.metrics import compute_metrics, METRIC_DECIMALS, METRICS
from kalematerials.utils.reporting import format_mean_std


def format_split_results(y_true, predictions: Mapping[str, np.ndarray]) -> str:
    """Format the regression metrics of one split, per model."""
    lines = ["Regression Metrics:"]
    for name, y_pred in predictions.items():
        metrics = compute_metrics(y_true, y_pred)
        lines.append(f"\n{name}:")
        lines.append(f"MSE: {metrics['mse']:.4f}")
        lines.append(f"MAE: {metrics['mae']:.4f}")
        lines.append(f"MRE: {metrics['mre']:.6f}")
        lines.append(f"R2:  {metrics['r2']:.4f}")
    return "\n".join(lines) + "\n"


def format_cross_validation_results(
    results: Mapping[str, Mapping[str, Sequence[float]]],
    title: str = "Cross-Validation Metrics (mean ± std):",
) -> str:
    """Format mean ± std of each metric over the repeats, per model."""

    def _std(values: Sequence[float]) -> float:
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    lines = [title]
    for name, scores in results.items():
        lines.append(f"\n{name}:")
        for metric in METRICS:
            values = list(scores[metric])
            label = "R2" if metric == "r2" else metric.upper()
            summary = format_mean_std(float(np.mean(values)), _std(values), METRIC_DECIMALS[metric])
            lines.append(f"{label + ':':<5}{summary}")
    return "\n".join(lines) + "\n"


def format_summary(summary: pd.DataFrame, title: str) -> str:
    """Format mean ± std per model and metric from a summary table."""
    if summary.empty:
        return f"\n{title}\n  (no observations)\n"

    lines = [f"\n{title}"]
    for model, group in summary.groupby("model", sort=True):
        lines.append(f"\n{model}:")
        for metric in METRICS:
            row = group[group["metric"] == metric]
            if row.empty:
                continue
            mean, std, n = float(row["mean"].iloc[0]), float(row["std"].iloc[0]), int(row["n"].iloc[0])
            label = "R2" if metric == "r2" else metric.upper()
            lines.append(f"{label + ':':<5}{format_mean_std(mean, std, METRIC_DECIMALS[metric])}  (n={n})")
    return "\n".join(lines) + "\n"


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse `<metric>_mean`/`<metric>_std` column pairs into "mean ± std" strings."""
    if df.empty:
        return df

    out = df.copy()
    for metric in METRICS:
        mean_col, std_col = f"{metric.upper()}_mean", f"{metric.upper()}_std"
        if mean_col not in out.columns or std_col not in out.columns:
            continue
        decimals = METRIC_DECIMALS[metric]
        out[metric.upper()] = [format_mean_std(mean, std, decimals) for mean, std in zip(out[mean_col], out[std_col])]
        out = out.drop(columns=[mean_col, std_col])

    return out
