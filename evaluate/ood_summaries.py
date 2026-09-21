"""Summaries an out-of-distribution run derives from its result and summary tables."""

from typing import Sequence

import numpy as np
import pandas as pd

from evaluate.metrics import COMPARABLE_METRICS
from evaluate.significance import MIN_PAIRS_FOR_TEST, compare_models_significance

# Split families, as they appear in the `split_type` column.
OOD = "OOD"
ID_PAIRED = "ID-paired"
ID_RANDOM = "ID-random"


def summarize_model_comparison(
    results: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    split_type: str = OOD,
    metrics: Sequence[str] = COMPARABLE_METRICS,
    min_pairs: int = MIN_PAIRS_FOR_TEST,
) -> pd.DataFrame:
    """Paired comparison of two models across the splits of each scenario.

    One paired observation per split, averaging over seeds first: seeds differ only in the inner K-fold and score the
    same test set, so treating them as separate observations would reintroduce the non-independence this exists to
    avoid.

    Args:
        results: Long-form result rows.
        model_a, model_b: Model names to compare.
        split_type: Which rows to compare on.
        metrics: Which metrics to test; lower-is-better only.
        min_pairs: Below this many splits no interval or p-value is reported.

    Returns:
        One row per (scenario, metric), or an empty frame if the models or split_type are absent.
    """
    subset = results[(results["split_type"] == split_type) & results["model"].isin([model_a, model_b])]
    if subset.empty or not {model_a, model_b}.issubset(set(subset["model"])):
        return pd.DataFrame()

    rows = []
    for scenario, group in subset.groupby("scenario", sort=True):
        wide = group.pivot_table(index="split_id", columns=["metric", "model"], values="value", aggfunc="mean")
        for metric in metrics:
            if (metric, model_a) not in wide.columns or (metric, model_b) not in wide.columns:
                continue
            pair = wide[[(metric, model_a), (metric, model_b)]].dropna()
            if pair.empty:
                continue

            paired = {
                model_a: {metric: pair[(metric, model_a)].tolist()},
                model_b: {metric: pair[(metric, model_b)].tolist()},
            }
            result = compare_models_significance(paired, model_a, model_b, metric=metric, min_pairs=min_pairs)
            rows.append(dict(
                scenario=scenario, metric=metric.upper(),
                model_a=model_a, model_b=model_b,
                n_splits=result.n_pairs,
                mean_difference=result.mean_difference,
                ci_low=result.ci_low,
                ci_high=result.ci_high,
                t_pvalue=result.t_pvalue,
                # An interval clear of zero is the same statement as p < 0.05, read off the quantity that was reported.
                significant=bool(np.isfinite(result.ci_low) and (result.ci_low > 0 or result.ci_high < 0)),
                note=result.note or "",
            ))

    return pd.DataFrame(rows)


def summarize_generalization_gap(summary: pd.DataFrame, metric: str = "mse") -> pd.DataFrame:
    """Decompose each model's drop under shift into a test-side part and a training-pool part.

    shift_gap        OOD - ID-paired. Same models and training rows, so this is purely the cost of testing on the
                     held-out region.
    train_pool_gap   ID-paired - ID-random. Same training-set size, but one pool has a whole chemistry family carved out
                     of it; this is what an unmatched ID-vs-OOD comparison silently folds into the shift number.

    Args:
        summary: The summary table, one mean per (scenario, split_type, model, metric).
        metric: Which metric to decompose; lower-is-better metrics only.

    Returns:
        One row per (scenario, model), or an empty frame if `summary` has no rows for `metric`.
    """
    rows = summary[summary["metric"] == metric]
    if rows.empty:
        return pd.DataFrame()

    wide = rows.pivot_table(index=["scenario", "model"], columns="split_type", values="mean").reset_index()
    for split_type in (OOD, ID_PAIRED, ID_RANDOM):
        if split_type not in wide.columns:
            wide[split_type] = np.nan
    wide["shift_gap"] = wide[OOD] - wide[ID_PAIRED]
    wide["train_pool_gap"] = wide[ID_PAIRED] - wide[ID_RANDOM]
    renamed = {split_type: f"{metric.upper()}_{split_type}" for split_type in (OOD, ID_PAIRED, ID_RANDOM)}
    return wide.rename(columns=renamed)[["scenario", "model", *renamed.values(), "shift_gap", "train_pool_gap"]]
