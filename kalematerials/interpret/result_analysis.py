"""Model comparisons and generalization gaps computed from saved result tables."""

from typing import Sequence

import numpy as np
import pandas as pd

from kalematerials.evaluate.metrics import COMPARABLE_METRICS
from kalematerials.interpret.significance import compare_models_significance, MIN_PAIRS_FOR_TEST
from kalematerials.loaddata.splits import ID_RANDOM, ID_REFERENCE, OOD


def _test_train_ratio(splits: pd.DataFrame, scenario: str) -> float:
    """Mean n_test / n_train over the scenario's splits, for the corrected t-test."""
    rows = splits[splits["scenario"] == scenario] if not splits.empty else splits
    if rows.empty:
        raise ValueError(f"No split sizes for scenario {scenario!r}; the comparison needs n_train and n_test.")
    return float((rows["n_test"] / rows["n_train"]).mean())


def summarize_model_comparison(
    results: pd.DataFrame,
    splits: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    metrics: Sequence[str] = COMPARABLE_METRICS,
    min_pairs: int = MIN_PAIRS_FOR_TEST,
) -> pd.DataFrame:
    """Compare two models per scenario, split type and lower-is-better metric.

    A scenario with OOD rows is compared on those alone: seeds share each held-out set, so they are averaged
    and the held-out sets are paired without a variance correction. Other split types pair every
    (seed, split_id) and correct for shared training rows with the mean n_test / n_train from `splits`,
    which for repeated K-fold gives the corrected repeated cross-validation test. Skip tests below `min_pairs`.
    Return one row per (scenario, split_type, metric), or an empty table if the models are absent.
    """
    subset = results[results["model"].isin([model_a, model_b])]
    if subset.empty or not {model_a, model_b}.issubset(set(subset["model"])):
        return pd.DataFrame()

    rows = []
    for scenario, group in subset.groupby("scenario", sort=True):
        present = set(group["split_type"])
        for split_type in [OOD] if OOD in present else sorted(present):
            part = group[group["split_type"] == split_type]
            if split_type == OOD:
                index, test_train_ratio = ["split_id"], 0.0
            else:
                index, test_train_ratio = ["seed", "split_id"], _test_train_ratio(splits, scenario)

            wide = part.pivot_table(index=index, columns=["metric", "model"], values="value", aggfunc="mean")
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
                result = compare_models_significance(
                    paired,
                    model_a,
                    model_b,
                    metric=metric,
                    min_pairs=min_pairs,
                    test_train_ratio=test_train_ratio,
                )
                rows.append(
                    dict(
                        scenario=scenario,
                        split_type=split_type,
                        metric=metric.upper(),
                        model_a=model_a,
                        model_b=model_b,
                        n_splits=result.n_pairs,
                        mean_difference=result.mean_difference,
                        ci_low=result.ci_low,
                        ci_high=result.ci_high,
                        t_pvalue=result.t_pvalue,
                        significant=bool(np.isfinite(result.ci_low) and (result.ci_low > 0 or result.ci_high < 0)),
                        note=result.note or "",
                    )
                )

    return pd.DataFrame(rows)


def summarize_generalization_gap(summary: pd.DataFrame, metric: str = "mse") -> pd.DataFrame:
    """Return shift_gap = OOD - ID-reference and train_pool_gap = ID-reference - ID-random per (scenario, model).

    Use a lower-is-better `metric` from the summary table. Return an empty table if the summary has no OOD
    rows for the metric.
    """
    rows = summary[(summary["metric"] == metric) & (summary["split_type"] == OOD)] if not summary.empty else summary
    if rows.empty:
        return pd.DataFrame()
    rows = summary[(summary["metric"] == metric) & summary["scenario"].isin(set(rows["scenario"]))]

    wide = rows.pivot_table(index=["scenario", "model"], columns="split_type", values="mean").reset_index()
    for split_type in (OOD, ID_REFERENCE, ID_RANDOM):
        if split_type not in wide.columns:
            wide[split_type] = np.nan
    wide["shift_gap"] = wide[OOD] - wide[ID_REFERENCE]
    wide["train_pool_gap"] = wide[ID_REFERENCE] - wide[ID_RANDOM]
    renamed = {split_type: f"{metric.upper()}_{split_type}" for split_type in (OOD, ID_REFERENCE, ID_RANDOM)}
    return wide.rename(columns=renamed)[["scenario", "model", *renamed.values(), "shift_gap", "train_pool_gap"]]
