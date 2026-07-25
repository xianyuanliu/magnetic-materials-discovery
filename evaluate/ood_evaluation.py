"""Per-split evaluation and table-building for OOD stress tests.

Given a set of (train_idx, test_idx) OOD splits (built by pipeline/ood_splits.py) and
orchestrated by pipeline/ood_pipeline.py, this module runs KFold on the TRAIN portion of
each split and scores against the fixed, held-out OOD TEST portion, then rolls the
results up into reportable tables.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from evaluate.metrics import compute_metrics, format_mean_std
from evaluate.cross_validation import compare_models_significance
from pipeline.train import DEFAULT_TUNE_CV_FOLDS, DEFAULT_TUNE_N_ITER

# Metric keys carried through Tables 2 and 4, and the decimals each is printed
# with. MRE is a small fraction, so it needs more places than the rest.
METRICS = ("mse", "mae", "mre", "r2")
METRIC_DECIMALS = {"mse": 4, "mae": 4, "mre": 6, "r2": 4}


def evaluate_splits_kfold_train_fixed_test(
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    model_keys,
    model_registry,
    *,
    scenario: str,
    seed: int,
    cv_folds: int,
    shuffle: bool,
    hyperparameter_tuning: bool,
    best_params: Optional[Dict] = None,
    model_random_state: int = 0,
    rf_name: Optional[str],
    xgb_name: Optional[str],
    tune_cv_folds: int = DEFAULT_TUNE_CV_FOLDS,
    tune_n_iter: int = DEFAULT_TUNE_N_ITER,
):
    """Run inner KFold on each split's TRAIN portion, score against its fixed OOD TEST.

    model_random_state seeds model construction and hyperparameter search
    (independent of `seed`, which seeds the inner KFold split on TRAIN).

    tune_cv_folds/tune_n_iter bound the hyperparameter search, which re-runs
    per fold per split per seed — the most expensive place in the codebase to
    leave the budget unbounded.
    """

    summary_rows = []
    metrics_rows = []
    signif_rows = []

    kf_seed = int(seed)

    for split_id, train_idx, test_idx in splits:

        X_train_full = X.iloc[train_idx]
        y_train_full = y.iloc[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        if len(train_idx) < cv_folds:
            print(
                f"[WARN] Skipping split {split_id} in {scenario}: "
                f"n_train={len(train_idx)} < cv_folds={cv_folds}"
            )
            continue

        kf = KFold(
            n_splits=cv_folds,
            shuffle=shuffle,
            random_state=kf_seed if shuffle else None,
        )

        per_fold = {}

        for key in model_keys:
            name = model_registry[key]["name"]
            per_fold[name] = {"mse": [], "mae": [], "mre": [], "r2": []}

        for train_sub_idx, _ in kf.split(X_train_full):

            X_tr = X_train_full.iloc[train_sub_idx]
            y_tr = y_train_full.iloc[train_sub_idx]

            for key in model_keys:

                model_cfg = model_registry[key]

                params = None
                if best_params is not None and key in best_params:
                    params = best_params[key]
                elif hyperparameter_tuning and model_cfg["tune"] is not None:
                    params = model_cfg["tune"](
                        X_tr,
                        y_tr,
                        cv_folds=tune_cv_folds,
                        random_state=model_random_state,
                        n_iter=tune_n_iter,
                    )

                model = model_cfg["train"](X_tr, y_tr, params=params, random_state=model_random_state)

                y_pred = model.predict(X_test)

                metrics = compute_metrics(y_test, y_pred)

                name = model_cfg["name"]

                for m in metrics:
                    per_fold[name][m].append(metrics[m])

        # Table 1
        summary_rows.append(
            dict(
                scenario=scenario,
                split_id=split_id,
                heldout_target=split_id.split("=")[-1],
                n_train=len(train_idx),
                n_test=len(test_idx),
                seed=seed,
            )
        )

        # Table 2 — numeric mean/std columns; formatting happens at print time
        # only, so downstream aggregation (Table 4) and the exported CSV stay
        # machine-readable at full precision.
        for model_name, scores in per_fold.items():

            row = dict(scenario=scenario, split_id=split_id, seed=seed, model=model_name)
            for metric in METRICS:
                values = scores[metric]
                row[f"{metric.upper()}_mean"] = float(np.mean(values))
                row[f"{metric.upper()}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
            row["n_test"] = len(test_idx)
            metrics_rows.append(row)

        # Table 3 — RF vs XGB
        if rf_name and xgb_name and rf_name in per_fold and xgb_name in per_fold:

            cv_like = {
                rf_name: per_fold[rf_name],
                xgb_name: per_fold[xgb_name],
            }

            for metric in ["mse", "mae"]:

                _, t_p, _, w_p = compare_models_significance(
                    cv_like,
                    rf_name,
                    xgb_name,
                    metric=metric,
                )

                signif_rows.append(
                    dict(
                        scenario=scenario,
                        split_id=split_id,
                        seed=seed,
                        metric=metric.upper(),
                        t_pvalue=t_p,
                        wilcoxon_pvalue=w_p,
                        significant=(t_p < 0.05) or (w_p < 0.05),
                    )
                )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(metrics_rows),
        pd.DataFrame(signif_rows),
    )


def summarize_runs_across_splits(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Average Table 2's per-split means into one row per (scenario, model).

    Reads the numeric `<METRIC>_mean` columns directly. It used to re-parse
    them out of "mean ± std" display strings, which truncated every value to
    the formatter's 4 decimals and turned a single NaN split into a silently
    NaN row for the whole scenario.

    The reported spread is the std *of the per-split means*, i.e. how much a
    model's score moves between OOD splits — not the within-split fold spread,
    which stays in Table 2. n_splits is how many splits the row summarizes; a
    NaN metric is dropped from that metric's mean only, and stays visible in
    Table 2 rather than propagating into everything.

    Args:
        metrics_df: Table 2 output of evaluate_splits_kfold_train_fixed_test,
            concatenated across splits/seeds.
    """

    rows = []

    if metrics_df.empty:
        return pd.DataFrame(rows)

    for (scenario, model), g in metrics_df.groupby(["scenario", "model"]):

        row = dict(scenario=scenario, model=model)

        for metric in METRICS:
            values = g[f"{metric.upper()}_mean"].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            row[f"{metric.upper()}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric.upper()}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )

        row["n_splits"] = len(g)
        rows.append(row)

    return pd.DataFrame(rows)


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


def print_ood_tables(table1, table2, table3, table4):
    """Print the 4 OOD result tables (per-split summary/metrics/significance/combined)."""

    print("\nTable 1: Scenario summary")
    print(table1.to_string(index=False))

    print("\nTable 2: Metrics by model")
    print(format_metric_table(table2).to_string(index=False))

    print("\nTable 3: RF vs XGB significance")
    print(table3.to_string(index=False))

    print("\nTable 4: Combined comparison")
    print(format_metric_table(table4).to_string(index=False))
