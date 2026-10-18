"""Progress logging and console output for result tables."""

import sys
from typing import Optional

import numpy as np
import pandas as pd


class Logger:
    """Write text to a stream; None silences output."""

    def __init__(self, stream=sys.stdout):
        self.stream = stream

    def write(self, message: str) -> None:
        """Write `message` verbatim, without adding a newline."""
        if self.stream is not None:
            self.stream.write(message)


SILENT = Logger(None)


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format a "mean ± std" string, or "nan" if either value is missing."""
    if mean is None or std is None or np.isnan(mean) or np.isnan(std):
        return "nan"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


RADIX_NAMES = {2: "binary", 3: "ternary", 4: "quaternary", 5: "quinary", 6: "senary", 7: "septenary"}


def print_compound_counts(counts: pd.Series) -> None:
    """Print the number of compounds per number of distinct elements."""
    for radix, count in counts.items():
        name = RADIX_NAMES.get(int(radix), f"{int(radix)}-component")
        print(f"We have {count} {name} compounds")


def format_search_budget(n_seeds: int, n_folds: int, n_tunable: int) -> str:
    """Format the number of nested hyperparameter searches a run will make."""
    return (
        f"\n[INFO] Nested hyperparameter search: {n_seeds} seed(s) x {n_folds} folds x "
        f"{n_tunable} tunable model(s) = {n_seeds * n_folds * n_tunable} searches.\n"
    )


def _print_frame(title: str, df: pd.DataFrame, note: Optional[str] = None) -> None:
    """Print one named result table, or a placeholder when it is empty."""
    print(f"\n{title}")
    if note:
        print(f"({note})")
    print(df.to_string(index=False) if not df.empty else "  (empty)")


def print_ood_tables(
    splits: pd.DataFrame,
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    generalization_gap: pd.DataFrame,
) -> None:
    """Print the OOD result tables, with the summary pivoted wide."""
    _print_frame("Splits", splits)
    wide = summary.pivot_table(index=["scenario", "split_type", "model"], columns="metric", values="mean")
    _print_frame("Summary (mean over splits and seeds)", wide.reset_index())
    _print_frame(
        "Model comparison significance",
        significance,
        note="paired across OOD splits — one observation per split, not per inner fold",
    )
    _print_frame("Generalization gap (MSE), shift vs training-pool", generalization_gap)


def print_analysis(significance: pd.DataFrame, generalization_gap: pd.DataFrame) -> None:
    """Print the nonempty analysis tables."""
    if not significance.empty:
        _print_frame("Model comparison significance", significance)
    if not generalization_gap.empty:
        _print_frame("Generalization gap (MSE), shift vs training-pool", generalization_gap)


def print_uncertainty_report(across_seeds: pd.DataFrame, alpha: float) -> None:
    """Print the calibration table across seeds."""
    nominal = 1.0 - alpha
    print(f"\nCalibration by split type and method (nominal coverage {nominal:.0%}, mean ± std over seeds)")
    print(across_seeds.to_string(index=False))


def print_predictions(predictions: pd.DataFrame, max_rows: int = 50) -> None:
    """Print a prediction table, truncating a long one."""
    shown = predictions.head(max_rows)
    print(shown.to_string(index=False))
    if len(predictions) > max_rows:
        print(f"... {len(predictions) - max_rows} more row(s) not shown")
