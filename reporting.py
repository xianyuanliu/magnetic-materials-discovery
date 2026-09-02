"""Console formatting for run results.

All printing lives here. The scoring modules under `evaluate/` return frames and
dataclasses, so they can be called from a notebook or another project without a
run's console output appearing as a side effect; this module turns those results
into the text a CLI run shows.
"""

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from core import METRIC_DECIMALS, METRICS
from evaluate.cross_validation import SignificanceResult, WinCounts
from evaluate.metrics import compute_metrics


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format a "mean ± std" string, or "nan" if either value is missing."""
    if mean is None or std is None or np.isnan(mean) or np.isnan(std):
        return "nan"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def print_holdout_results(y_true, predictions: Mapping[str, np.ndarray]) -> None:
    """Print the regression metrics for one holdout split, per model."""
    print("Regression Metrics:")
    for name, y_pred in predictions.items():
        metrics = compute_metrics(y_true, y_pred)
        print(f"\n{name}:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MRE: {metrics['mre']:.6f}")
        print(f"R2:  {metrics['r2']:.4f}")


def print_cv_results(
    results: Mapping[str, Mapping[str, Sequence[float]]],
    title: str = "Cross-Validation Metrics (mean ± std):",
) -> None:
    """Print mean ± std metrics across repeats, for CV folds or holdout splits."""
    print(title)

    def _std(values: Sequence[float]) -> float:
        # ddof=1 is undefined for a single repeat (one holdout split, say).
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    for name, scores in results.items():
        print(f"\n{name}:")
        for metric in METRICS:
            values = list(scores[metric])
            label = "R2" if metric == "r2" else metric.upper()
            summary = format_mean_std(
                float(np.mean(values)), _std(values), METRIC_DECIMALS[metric]
            )
            print(f"{label + ':':<5}{summary}")


def print_significance(result: SignificanceResult) -> None:
    """Print one paired model comparison, including why a p-value is missing."""
    print(f"\n{result.model_a} vs {result.model_b} — metric={result.metric.upper()}")
    print(f"  paired observations: {result.n_pairs}")
    print(f"  mean difference (a - b): {result.mean_difference:+.6g}")
    if result.note:
        print(f"  no p-value reported: {result.note}")
        return
    print(f"  paired t-test:        t={result.t_stat:.4f}, p={result.t_pvalue:.6g}")
    print(f"  Wilcoxon signed-rank: W={result.w_stat:.4f}, p={result.w_pvalue:.6g}")


def print_win_counts(wins: WinCounts, total: int) -> None:
    """Print the fold-by-fold win tally between two models."""
    print(f"\nFold-by-fold win count (metric={wins.metric.upper()}): "
          f"{wins.model_a} vs {wins.model_b}")
    width = max(len(wins.model_a), len(wins.model_b), len("Ties")) + 7
    print(f"{wins.model_a + ' wins:':<{width}}{wins.a_wins}/{total}")
    print(f"{wins.model_b + ' wins:':<{width}}{wins.b_wins}/{total}")
    print(f"{'Ties:':<{width}}{wins.ties}/{total}")


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse `<METRIC>_mean`/`<METRIC>_std` column pairs into display strings.

    Display only — the underlying frames keep full-precision numbers so the
    saved CSVs stay usable for arithmetic.
    """
    if df.empty:
        return df

    out = df.copy()
    for metric in METRICS:
        mean_col, std_col = f"{metric.upper()}_mean", f"{metric.upper()}_std"
        if mean_col not in out.columns or std_col not in out.columns:
            continue
        decimals = METRIC_DECIMALS[metric]
        out[metric.upper()] = [
            format_mean_std(mean, std, decimals)
            for mean, std in zip(out[mean_col], out[std_col])
        ]
        out = out.drop(columns=[mean_col, std_col])

    return out


def _print_frame(title: str, df: pd.DataFrame, note: Optional[str] = None) -> None:
    """Print one named result table, or a placeholder when it is empty."""
    print(f"\n{title}")
    if note:
        print(f"({note})")
    print(df.to_string(index=False) if not df.empty else "  (empty)")


def print_ood_tables(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    table5: pd.DataFrame,
) -> None:
    """Print the OOD result tables."""
    _print_frame("Table 1: Scenario summary", table1)
    _print_frame("Table 2: Metrics by model", format_metric_table(table2))
    _print_frame(
        "Table 3: Model comparison significance",
        table3,
        note="paired across OOD splits — one observation per split, not per inner fold",
    )
    _print_frame("Table 4: Combined comparison", format_metric_table(table4))
    _print_frame("Table 5: Generalisation gap (MSE), shift vs training-pool", table5)


def print_uq_report(across_seeds: pd.DataFrame, alpha: float) -> None:
    """Print the headline calibration comparison without editorialising it."""
    nominal = 1.0 - alpha
    print(
        f"\nCalibration by split type and method "
        f"(nominal coverage {nominal:.0%}, mean ± std over seeds)"
    )
    print(across_seeds.to_string(index=False))
    print(
        "\nHow to read it: coverage below nominal means the interval is too narrow "
        "(overconfident); rms_z above 1 means the same for the sigma itself. "
        "Compare OOD against ID-random, not ID-kfold — only ID-random holds the "
        "training-set size fixed."
    )


def print_predictions(predictions: pd.DataFrame, max_rows: int = 50) -> None:
    """Print a prediction table, truncating a long one."""
    shown = predictions.head(max_rows)
    print(shown.to_string(index=False))
    if len(predictions) > max_rows:
        print(f"... {len(predictions) - max_rows} more row(s) not shown")
