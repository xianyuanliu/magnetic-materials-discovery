"""Calibration metrics for predictive uncertainty.

Scores a prediction interval against what actually happened: does it cover the
truth as often as it claims, and how wide does it have to be to manage that.

Everything here aggregates over *samples*, never over splits. Mixing the two —
averaging per-split errors while pooling per-sample uncertainties — makes the
two halves of any error-vs-uncertainty comparison incommensurable as soon as
the splits differ in size, which LOCO clusters always do.
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from core import DEFAULT_ALPHA

# Columns a per-sample prediction frame must carry for the summaries below.
SAMPLE_COLUMNS = ("y_true", "y_pred", "sigma", "half_width")


def compute_calibration_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    half_width: Sequence[float],
    sigma: Sequence[float],
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, float]:
    """Score one pooled set of predictions against its uncertainty estimate.

    Args:
        y_true: Observed targets.
        y_pred: Point predictions.
        half_width: Half-width of each prediction interval, so the interval is
            y_pred ± half_width.
        sigma: Per-sample standard-deviation estimate behind the interval.
        alpha: Nominal miscoverage the interval was built for.

    Returns:
        mae, rmse: Point-prediction accuracy.
        mean_sigma: Average claimed uncertainty.
        coverage: Fraction of samples the interval actually contains.
        coverage_error: coverage minus its nominal (1 - alpha) target. Negative
            means the interval is too narrow, i.e. the model is overconfident.
        mean_width: Average interval width — sharpness. Coverage alone is
            trivially satisfied by a wide enough interval, so the pair is what
            carries the information.
        mean_abs_z, rms_z: |error| / sigma. A calibrated Gaussian sigma gives
            rms_z = 1; above 1 is overconfident, below 1 conservative. These are
            the scale-sensitive numbers, so they detect the miscalibration that
            any max-normalised score would divide away.
        n: Sample count behind the row, so the pooling stays auditable.
        n_zero_sigma: How many of those samples were left out of the z columns.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    half_width = np.asarray(half_width, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    error = np.abs(y_pred - y_true)
    coverage = float(np.mean(error <= half_width))

    # A sample where every tree agrees has sigma = 0 and no finite z. Those are
    # dropped from the z columns only — one of them would otherwise send the
    # whole group's rms_z to infinity — and counted so the omission is visible.
    scored = sigma > 0
    z = error[scored] / sigma[scored]

    return {
        "n": int(error.size),
        "n_zero_sigma": int((~scored).sum()),
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean_sigma": float(np.mean(sigma)),
        "coverage": coverage,
        "coverage_error": coverage - (1.0 - alpha),
        "mean_width": float(np.mean(2.0 * half_width)),
        "mean_abs_z": float(np.mean(z)) if z.size else np.nan,
        "rms_z": float(np.sqrt(np.mean(z ** 2))) if z.size else np.nan,
    }


def summarize_calibration(
    samples: pd.DataFrame,
    by: Sequence[str],
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Pool a per-sample prediction frame into one calibration row per group.

    Args:
        samples: One row per predicted sample, with the SAMPLE_COLUMNS plus
            whatever grouping columns `by` names.
        by: Grouping columns, e.g. ("split_type", "method").
        alpha: Nominal miscoverage.

    Returns:
        One row per group, sorted by the grouping columns.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = [c for c in (*SAMPLE_COLUMNS, *by) if c not in samples.columns]
    if missing:
        raise ValueError(f"Missing columns in per-sample frame: {missing}")

    rows: List[Dict] = []
    for keys, group in samples.groupby(list(by), sort=True):
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row.update(compute_calibration_metrics(
            group["y_true"], group["y_pred"], group["half_width"], group["sigma"], alpha,
        ))
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_across_seeds(
    per_seed: pd.DataFrame,
    by: Sequence[str],
    metrics: Sequence[str] = ("mae", "coverage", "mean_sigma", "mean_width", "rms_z"),
) -> pd.DataFrame:
    """Collapse per-seed calibration rows into mean ± std across seeds.

    The spread here is the only honest error bar on these numbers: a single seed
    moves coverage and the error/uncertainty ratio by more than the differences
    being claimed.

    Args:
        per_seed: Output of summarize_calibration grouped with `seed` in `by`.
        by: Grouping columns to keep, excluding `seed`.
        metrics: Which metric columns to aggregate.
    """
    if per_seed.empty:
        return pd.DataFrame()

    available = [m for m in metrics if m in per_seed.columns]
    summary = per_seed.groupby(list(by), sort=True)[available].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = per_seed.groupby(list(by), sort=True).size()

    return summary.reset_index().fillna({f"{m}_std": 0.0 for m in available})
