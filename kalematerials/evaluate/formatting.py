"""Console formatting for regression metrics, summaries and significance tests."""

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from kalematerials.evaluate.metrics import COMPARABLE_METRICS, compute_metrics, METRIC_DECIMALS, METRICS
from kalematerials.evaluate.significance import compare_models_significance, SignificanceResult
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


def format_comparisons(
    results,
    model_a: str,
    model_b: str,
    metrics: Sequence[str] = COMPARABLE_METRICS,
    test_train_ratio: float = 0.0,
) -> str:
    """Format paired significance tests for two models on lower-is-better metrics.

    Scores and `test_train_ratio` follow evaluate.significance.compare_models_significance.
    """
    return "".join(
        format_significance(
            compare_models_significance(
                results,
                model_a,
                model_b,
                metric=metric,
                test_train_ratio=test_train_ratio,
            )
        )
        for metric in metrics
    )


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


def format_significance(result: SignificanceResult) -> str:
    """Format one paired model comparison."""
    lines = [
        f"\n{result.model_a} vs {result.model_b} — metric={result.metric.upper()}",
        f"  paired observations: {result.n_pairs}",
    ]

    if np.isnan(result.ci_low):
        lines.append(f"  difference (a - b): {result.mean_difference:+.6g}")
    else:
        lines.append(
            f"  difference (a - b): {result.mean_difference:+.6g}"
            f"  95% CI [{result.ci_low:+.6g}, {result.ci_high:+.6g}]"
        )

    if np.isnan(result.t_pvalue):
        lines.append(f"  no p-value reported: {result.note}")
    else:
        lines.append(f"  paired t-test: t={result.t_stat:.4f}, p={result.t_pvalue:.6g}")
        if result.note:
            lines.append(f"  note: {result.note}")

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
