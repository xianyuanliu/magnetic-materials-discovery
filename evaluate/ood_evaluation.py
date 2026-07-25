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

        # Table 2
        for model_name, scores in per_fold.items():

            metrics_rows.append(
                dict(
                    scenario=scenario,
                    split_id=split_id,
                    seed=seed,
                    model=model_name,
                    MSE=format_mean_std(
                        np.mean(scores["mse"]),
                        np.std(scores["mse"], ddof=1) if len(scores["mse"]) > 1 else 0.0,
                    ),
                    MAE=format_mean_std(
                        np.mean(scores["mae"]),
                        np.std(scores["mae"], ddof=1) if len(scores["mae"]) > 1 else 0.0,
                    ),
                    MRE=format_mean_std(
                        np.mean(scores["mre"]),
                        np.std(scores["mre"], ddof=1) if len(scores["mre"]) > 1 else 0.0,
                        6,
                    ),
                    R2=format_mean_std(
                        np.mean(scores["r2"]),
                        np.std(scores["r2"], ddof=1) if len(scores["r2"]) > 1 else 0.0,
                    ),
                    n_test=len(test_idx),
                )
            )

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
    """Average per-split "mean ± std" metrics into one row per (scenario, model).

    Args:
        metrics_df: Table 2 output of evaluate_splits_kfold_train_fixed_test,
            concatenated across splits/seeds.
    """

    rows = []

    if metrics_df.empty:
        return pd.DataFrame(rows)

    def parse_mean(x):
        return float(str(x).split("±")[0].strip())

    for (scenario, model), g in metrics_df.groupby(["scenario", "model"]):

        mse = g["MSE"].map(parse_mean).to_numpy()
        mae = g["MAE"].map(parse_mean).to_numpy()
        mre = g["MRE"].map(parse_mean).to_numpy()
        r2 = g["R2"].map(parse_mean).to_numpy()

        rows.append(
            dict(
                scenario=scenario,
                model=model,
                MSE=format_mean_std(np.mean(mse), np.std(mse, ddof=1) if len(mse) > 1 else 0.0),
                MAE=format_mean_std(np.mean(mae), np.std(mae, ddof=1) if len(mae) > 1 else 0.0),
                MRE=format_mean_std(np.mean(mre), np.std(mre, ddof=1) if len(mre) > 1 else 0.0, 6),
                R2=format_mean_std(np.mean(r2), np.std(r2, ddof=1) if len(r2) > 1 else 0.0),
                n_splits=len(g),
            )
        )

    return pd.DataFrame(rows)


def print_ood_tables(table1, table2, table3, table4):
    """Print the 4 OOD result tables (per-split summary/metrics/significance/combined)."""

    print("\nTable 1: Scenario summary")
    print(table1.to_string(index=False))

    print("\nTable 2: Metrics by model")
    print(table2.to_string(index=False))

    print("\nTable 3: RF vs XGB significance")
    print(table3.to_string(index=False))

    print("\nTable 4: Combined comparison")
    print(table4.to_string(index=False))
