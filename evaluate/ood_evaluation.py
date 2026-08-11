"""Per-split evaluation and table-building for OOD stress tests.

Given a set of (train_idx, test_idx) OOD splits (built by pipeline/ood_splits.py) and
orchestrated by pipeline/ood_pipeline.py, this module runs KFold on the TRAIN portion of
each split and scores against the fixed, held-out OOD TEST portion.

Every OOD split is scored next to two in-distribution references, so a drop can
be attributed instead of merely observed:

  ID-paired   The same fold models, scored on the inner validation fold they
              already held out. Identical training rows, in-distribution test,
              so OOD minus ID-paired isolates the *test-side* shift.
  ID-random   A random train/test split of the same two sizes. Holding out Fe
              costs Novamag more than half its training data, so this is what
              separates "never saw Fe" from "trained on half as much".
"""

import zlib
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from evaluate.metrics import compute_metrics, format_mean_std
from evaluate.cross_validation import compare_models_significance
from pipeline.ood_splits import build_size_matched_split
from pipeline.train import DEFAULT_TUNE_CV_FOLDS, DEFAULT_TUNE_N_ITER

# Metric keys carried through Tables 2 and 4, and the decimals each is printed
# with. MRE is a small fraction, so it needs more places than the rest.
METRICS = ("mse", "mae", "mre", "r2")
METRIC_DECIMALS = {"mse": 4, "mae": 4, "mre": 6, "r2": 4}

# Values of the `split_type` column that distinguishes a shifted test set from
# its two in-distribution references (see the module docstring).
OOD = "OOD"
ID_PAIRED = "ID-paired"
ID_RANDOM = "ID-random"


@dataclass
class _FoldScores:
    """Per-inner-fold metrics for one evaluated test set."""

    per_model: Dict[str, Dict[str, List[float]]]
    n_test: int


def _empty_scores(model_keys, model_registry) -> Dict[str, Dict[str, List[float]]]:
    return {
        model_registry[key]["name"]: {metric: [] for metric in METRICS}
        for key in model_keys
    }


def _resolve_params(model_cfg, key, X_fit, y_fit, best_params, hyperparameter_tuning,
                    model_random_state, tune_cv_folds, tune_n_iter) -> Optional[Dict]:
    """Pick fixed parameters, or search for them inside this fold."""
    if best_params is not None and key in best_params:
        return best_params[key]
    if hyperparameter_tuning and model_cfg["tune"] is not None:
        return model_cfg["tune"](
            X_fit, y_fit, cv_folds=tune_cv_folds,
            random_state=model_random_state, n_iter=tune_n_iter,
        )
    return None


def _score_fold_models(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_keys,
    model_registry,
    *,
    kf: KFold,
    score_inner: bool,
    hyperparameter_tuning: bool,
    best_params: Optional[Dict],
    model_random_state: int,
    tune_cv_folds: int,
    tune_n_iter: int,
) -> Tuple[_FoldScores, Optional[_FoldScores]]:
    """Fit one model per inner fold and score it on the split's fixed test set.

    The inner KFold exists to average over training subsamples, not to select
    anything: no fold's score feeds back into fitting.

    Args:
        score_inner: Also score every fold model on the inner validation fold it
            held out. Free — the models are already fitted — and it is the only
            reference trained on exactly the same rows as the OOD score.

    Returns:
        (test_scores, inner_scores); inner_scores is None when score_inner is False.
    """
    X_train_full, y_train_full = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    test_scores = _empty_scores(model_keys, model_registry)
    inner_scores = _empty_scores(model_keys, model_registry) if score_inner else None
    inner_sizes: List[int] = []

    for fit_idx, inner_idx in kf.split(X_train_full):
        X_fit, y_fit = X_train_full.iloc[fit_idx], y_train_full.iloc[fit_idx]
        X_inner, y_inner = X_train_full.iloc[inner_idx], y_train_full.iloc[inner_idx]
        inner_sizes.append(len(inner_idx))

        for key in model_keys:
            model_cfg = model_registry[key]
            params = _resolve_params(
                model_cfg, key, X_fit, y_fit, best_params, hyperparameter_tuning,
                model_random_state, tune_cv_folds, tune_n_iter,
            )
            model = model_cfg["train"](X_fit, y_fit, params=params, random_state=model_random_state)

            name = model_cfg["name"]
            for metric, value in compute_metrics(y_test, model.predict(X_test)).items():
                test_scores[name][metric].append(value)
            if score_inner:
                for metric, value in compute_metrics(y_inner, model.predict(X_inner)).items():
                    inner_scores[name][metric].append(value)

    return (
        _FoldScores(test_scores, len(test_idx)),
        _FoldScores(inner_scores, int(round(float(np.mean(inner_sizes))))) if score_inner else None,
    )


def _metric_rows(scenario, split_id, seed, split_type, scores: _FoldScores) -> List[Dict]:
    """Table 2 rows: numeric mean/std per model, formatted only at print time."""
    rows = []
    for model_name, per_metric in scores.per_model.items():
        row = dict(
            scenario=scenario, split_id=split_id, seed=seed,
            split_type=split_type, model=model_name,
        )
        for metric in METRICS:
            values = per_metric[metric]
            row[f"{metric.upper()}_mean"] = float(np.mean(values))
            row[f"{metric.upper()}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["n_test"] = scores.n_test
        rows.append(row)
    return rows


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
    size_matched_control: bool = True,
):
    """Score each split's fixed OOD test set plus its in-distribution references.

    model_random_state seeds model construction and hyperparameter search
    (independent of `seed`, which seeds the inner KFold split on TRAIN).

    tune_cv_folds/tune_n_iter bound the hyperparameter search, which re-runs
    per fold per split per seed — the most expensive place in the codebase to
    leave the budget unbounded.

    Args:
        size_matched_control: Also evaluate a random split of the same train and
            test sizes (ID-random). Doubles the fits, and is what makes the
            reported degradation separable from the lost training data.

    Returns:
        (table1, table2, table3) — split summary, metrics by model and
        split_type, and the RF-vs-XGB significance rows (OOD only).
    """
    summary_rows, metrics_rows, signif_rows = [], [], []

    for split_id, train_idx, test_idx in splits:
        if len(train_idx) < cv_folds:
            print(
                f"[WARN] Skipping split {split_id} in {scenario}: "
                f"n_train={len(train_idx)} < cv_folds={cv_folds}"
            )
            continue

        kf = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=seed if shuffle else None)
        score = partial(
            _score_fold_models,
            X, y, model_keys=model_keys, model_registry=model_registry, kf=kf,
            hyperparameter_tuning=hyperparameter_tuning, best_params=best_params,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds, tune_n_iter=tune_n_iter,
        )

        ood_scores, id_paired_scores = score(train_idx, test_idx, score_inner=True)

        summary_rows.append(dict(
            scenario=scenario, split_id=split_id,
            heldout_target=split_id.split("=")[-1],
            n_train=len(train_idx), n_test=len(test_idx), seed=seed,
        ))

        metrics_rows += _metric_rows(scenario, split_id, seed, OOD, ood_scores)
        metrics_rows += _metric_rows(scenario, split_id, seed, ID_PAIRED, id_paired_scores)

        if size_matched_control:
            # crc32, not hash(): str hashing is salted per process, and this
            # seed has to be reproducible across runs.
            control_seed = zlib.crc32(f"{seed}|{scenario}|{split_id}".encode())
            control = build_size_matched_split(
                len(X), len(train_idx), len(test_idx), seed=control_seed,
            )
            if control is None:
                print(f"[WARN] No size-matched control fits for {scenario} {split_id}")
            else:
                _, control_train_idx, control_test_idx = control
                control_scores, _ = score(control_train_idx, control_test_idx, score_inner=False)
                metrics_rows += _metric_rows(scenario, split_id, seed, ID_RANDOM, control_scores)

        # Table 3 — RF vs XGB, on the OOD test set only.
        per_model = ood_scores.per_model
        if rf_name and xgb_name and rf_name in per_model and xgb_name in per_model:
            for metric in ("mse", "mae"):
                _, t_p, _, w_p = compare_models_significance(
                    {rf_name: per_model[rf_name], xgb_name: per_model[xgb_name]},
                    rf_name, xgb_name, metric=metric,
                )
                signif_rows.append(dict(
                    scenario=scenario, split_id=split_id, seed=seed,
                    metric=metric.upper(), t_pvalue=t_p, wilcoxon_pvalue=w_p,
                    significant=(t_p < 0.05) or (w_p < 0.05),
                ))

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(metrics_rows),
        pd.DataFrame(signif_rows),
    )


def summarize_runs_across_splits(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Average Table 2's per-split means into one row per (scenario, split_type, model).

    Reads the numeric `<METRIC>_mean` columns directly. It used to re-parse
    them out of "mean ± std" display strings, which truncated every value to
    the formatter's 4 decimals and turned a single NaN split into a silently
    NaN row for the whole scenario.

    The reported spread is the std *of the per-split means*, i.e. how much a
    model's score moves between OOD splits — not the within-split fold spread,
    which stays in Table 2. n_splits is how many rows the summary covers; a
    NaN metric is dropped from that metric's mean only, and stays visible in
    Table 2 rather than propagating into everything.

    Args:
        metrics_df: Table 2 output of evaluate_splits_kfold_train_fixed_test,
            concatenated across splits/seeds.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    group_cols = [c for c in ("scenario", "split_type", "model") if c in metrics_df.columns]
    rows = []

    for keys, group in metrics_df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))

        for metric in METRICS:
            values = group[f"{metric.upper()}_mean"].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            row[f"{metric.upper()}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric.upper()}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )

        row["n_splits"] = len(group)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_generalisation_gap(summary_df: pd.DataFrame, metric: str = "mse") -> pd.DataFrame:
    """Decompose each scenario's degradation into a shift part and a training-pool part.

    shift_gap        OOD - ID-paired. Same models and training rows, so this is
                     purely the cost of testing on the held-out region.
    train_pool_gap   ID-paired - ID-random. Same training-set size, but one pool
                     has a whole chemistry family carved out of it; this is what
                     an unmatched ID-vs-OOD comparison silently folds into the
                     shift number.

    Args:
        summary_df: Output of summarize_runs_across_splits.
        metric: Which metric to decompose; lower-is-better metrics only.

    Returns:
        One row per (scenario, model), or an empty frame if `summary_df` lacks
        the split_type column.
    """
    column = f"{metric.upper()}_mean"
    if summary_df.empty or "split_type" not in summary_df.columns:
        return pd.DataFrame()

    wide = summary_df.pivot_table(
        index=["scenario", "model"], columns="split_type", values=column,
    ).reset_index()

    for split_type in (OOD, ID_PAIRED, ID_RANDOM):
        if split_type not in wide.columns:
            wide[split_type] = np.nan

    wide["shift_gap"] = wide[OOD] - wide[ID_PAIRED]
    wide["train_pool_gap"] = wide[ID_PAIRED] - wide[ID_RANDOM]

    renamed = {split_type: f"{metric.upper()}_{split_type}" for split_type in (OOD, ID_PAIRED, ID_RANDOM)}
    return wide.rename(columns=renamed)[
        ["scenario", "model", *renamed.values(), "shift_gap", "train_pool_gap"]
    ]


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


def print_ood_tables(table1, table2, table3, table4, table5):
    """Print the OOD result tables (split summary/metrics/significance/combined/gap)."""
    print("\nTable 1: Scenario summary")
    print(table1.to_string(index=False))

    print("\nTable 2: Metrics by model")
    print(format_metric_table(table2).to_string(index=False))

    print("\nTable 3: RF vs XGB significance")
    print(table3.to_string(index=False))

    print("\nTable 4: Combined comparison")
    print(format_metric_table(table4).to_string(index=False))

    print("\nTable 5: Generalisation gap (MSE), shift vs training-pool")
    print(table5.to_string(index=False))
