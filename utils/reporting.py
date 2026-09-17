"""Console formatting for run results.

All printing lives here. The scoring modules under `evaluate/` return frames and dataclasses, so they can be called from
a notebook or another project without a run's console output appearing as a side effect; this module turns those results
into the text a CLI run shows.
"""

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from evaluate.cross_validation import SignificanceResult, compare_models_significance
from evaluate.metrics import COMPARABLE_METRICS, METRICS, METRIC_DECIMALS, compute_metrics


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
            summary = format_mean_std(float(np.mean(values)), _std(values), METRIC_DECIMALS[metric])
            print(f"{label + ':':<5}{summary}")


def print_comparisons(
    results,
    model_a: str,
    model_b: str,
    metrics: Sequence[str] = COMPARABLE_METRICS,
    test_train_ratio: float = 0.0,
) -> None:
    """Print one paired significance test per metric for a pair of models.

    Args:
        results: {model name: {metric: [one score per observation]}}.
        model_a, model_b: Model names to compare.
        metrics: Which metrics to test; lower-is-better only.
        test_train_ratio: Passed through so the t-test can correct for observations that share training data.
    """
    for metric in metrics:
        print_significance(compare_models_significance(
            results, model_a, model_b, metric=metric, test_train_ratio=test_train_ratio,
        ))


# Latin names for the first few, which read better than "3-component" in a dataset summary.
RADIX_NAMES = {2: "binary", 3: "ternary", 4: "quaternary", 5: "quinary", 6: "senary", 7: "septenary"}


def print_compound_counts(counts: pd.Series) -> None:
    """Print how many compounds have each number of distinct elements.

    Args:
        counts: Counts indexed by radix, from interpret.visualize.count_compounds_by_radix.
    """
    for radix, count in counts.items():
        name = RADIX_NAMES.get(int(radix), f"{int(radix)}-component")
        print(f"We have {count} {name} compounds")


def print_search_budget(
    n_seeds: int,
    n_folds: int,
    n_tunable: int,
    n_iter: int,
    tune_cv_folds: int,
) -> None:
    """Print what a nested hyperparameter search is about to cost, before it starts.

    The search re-runs inside every outer fold, so the fit count is a product of five numbers and reaches the hours
    long before any of them looks large on its own.

    Args:
        n_seeds, n_folds: Outer repeats and folds per repeat.
        n_tunable: Models that actually have something to search.
        n_iter, tune_cv_folds: Candidates sampled and inner folds per search.
    """
    searches = n_seeds * n_folds * n_tunable
    print(
        f"\n[INFO] Nested hyperparameter search: {n_seeds} seed(s) x {n_folds} folds x "
        f"{n_tunable} tunable model(s) = {searches} searches, each up to {n_iter} candidates x "
        f"{tune_cv_folds} inner folds (~{searches * n_iter * tune_cv_folds} model fits). "
        f"Lower tune_n_iter / tune_cv_folds in the config to shrink this."
    )


def print_summary(summary: pd.DataFrame, title: str) -> None:
    """Print a summary table from utils.persistence.save_results as mean ± std per model.

    Args:
        summary: Rows of (identifying columns..., mean, std, n).
        title: Heading, which names what the observations were.
    """
    if summary.empty:
        print(f"\n{title}\n  (no observations)")
        return

    print(f"\n{title}")
    for model, group in summary.groupby("model", sort=True):
        print(f"\n{model}:")
        for metric in METRICS:
            row = group[group["metric"] == metric]
            if row.empty:
                continue
            mean, std, n = float(row["mean"].iloc[0]), float(row["std"].iloc[0]), int(row["n"].iloc[0])
            label = "R2" if metric == "r2" else metric.upper()
            print(f"{label + ':':<5}{format_mean_std(mean, std, METRIC_DECIMALS[metric])}  (n={n})")


def print_significance(result: SignificanceResult) -> None:
    """Print one paired model comparison, including why a p-value is missing."""
    print(f"\n{result.model_a} vs {result.model_b} — metric={result.metric.upper()}")
    print(f"  paired observations: {result.n_pairs}")

    # The difference and its interval lead because they are in the metric's own units: they say by how much and how
    # precisely, where a p-value only says whether zero is excluded.
    if np.isnan(result.ci_low):
        print(f"  difference (a - b): {result.mean_difference:+.6g}")
    else:
        print(
            f"  difference (a - b): {result.mean_difference:+.6g}"
            f"  95% CI [{result.ci_low:+.6g}, {result.ci_high:+.6g}]"
        )

    # A note can mean either "the test did not run" or "it ran, read it with care", so the p-values decide which.
    if np.isnan(result.t_pvalue):
        print(f"  no p-value reported: {result.note}")
        return

    print(f"  paired t-test: t={result.t_stat:.4f}, p={result.t_pvalue:.6g}")
    if result.note:
        print(f"  note: {result.note}")


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse `<METRIC>_mean`/`<METRIC>_std` column pairs into display strings.

    Display only — the underlying frames keep full-precision numbers so the saved CSVs stay usable for arithmetic.
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
    splits_summary: pd.DataFrame,
    metrics_by_model: pd.DataFrame,
    comparison_significance: pd.DataFrame,
    combined_comparison: pd.DataFrame,
    generalization_gap: pd.DataFrame,
) -> None:
    """Print the OOD result tables."""
    _print_frame("Table 1: Scenario summary", splits_summary)
    _print_frame("Table 2: Metrics by model", format_metric_table(metrics_by_model))
    _print_frame(
        "Table 3: Model comparison significance",
        comparison_significance,
        note="paired across OOD splits — one observation per split, not per inner fold",
    )
    _print_frame("Table 4: Combined comparison", format_metric_table(combined_comparison))
    _print_frame("Table 5: Generalization gap (MSE), shift vs training-pool", generalization_gap)


def print_uq_report(across_seeds: pd.DataFrame, alpha: float) -> None:
    """Print the headline calibration comparison without editorializing it."""
    nominal = 1.0 - alpha
    print(f"\nCalibration by split type and method (nominal coverage {nominal:.0%}, mean ± std over seeds)")
    print(across_seeds.to_string(index=False))
    print(
        "\nHow to read it: coverage below nominal means the interval is too narrow (overconfident); rms_z above 1 "
        "means the same for the sigma itself. Compare OOD against ID-random, not ID-kfold — only ID-random holds the "
        "training-set size fixed."
    )


def print_predictions(predictions: pd.DataFrame, max_rows: int = 50) -> None:
    """Print a prediction table, truncating a long one."""
    shown = predictions.head(max_rows)
    print(shown.to_string(index=False))
    if len(predictions) > max_rows:
        print(f"... {len(predictions) - max_rows} more row(s) not shown")
