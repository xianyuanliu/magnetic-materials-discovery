"""Per-split scoring and table-building for OOD stress tests.

Given a set of (split_id, train_idx, test_idx) splits — built by
pipeline/ood_splits.py and orchestrated by pipeline/ood_pipeline.py — this runs
KFold on the TRAIN portion of each split and scores against the fixed, held-out
OOD TEST portion.

Every OOD split is scored next to two in-distribution references, so a drop can
be attributed instead of merely observed:

  ID-paired   The same fold models, scored on the inner validation fold they
              already held out. Identical training rows, in-distribution test,
              so OOD minus ID-paired isolates the *test-side* shift.
  ID-random   A random train/test split of the same two sizes, supplied by the
              caller. Holding out Fe costs Novamag more than half its training
              data, so this is what separates "never saw Fe" from "trained on
              half as much".

This module never prints and never imports from `pipeline`; it scores the splits
it is handed and returns frames. Reporting lives in reporting.py.
"""

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from utils.core import METRICS, Split
from utils.model_spec import DEFAULT_TUNE_CV_FOLDS, DEFAULT_TUNE_N_ITER, ModelSpec
from evaluate.cross_validation import MIN_PAIRS_FOR_TEST, compare_models_significance
from evaluate.metrics import compute_metrics

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


def _empty_scores(specs: Sequence[ModelSpec]) -> Dict[str, Dict[str, List[float]]]:
    return {spec.name: {metric: [] for metric in METRICS} for spec in specs}


def heldout_target(split_id: str) -> str:
    """Extract the held-out target from a split id, or "" when there isn't one.

    Membership splits are named `E=Fe`, `P=4`, `G=8`, `C=3`; the sparsity
    families are named `SparseX_top10pct` and hold out a *fraction*, not a
    target. Splitting unconditionally on "=" used to make those rows repeat the
    whole split id in a column meant for a chemistry label.
    """
    return split_id.split("=", 1)[1] if "=" in split_id else ""


def _resolve_params(
    spec: ModelSpec, X_fit, y_fit, best_params, hyperparameter_tuning,
    model_random_state, tune_cv_folds, tune_n_iter,
) -> Optional[Dict]:
    """Pick fixed parameters, or search for them inside this fold."""
    if best_params is not None and spec.key in best_params:
        return best_params[spec.key]
    if hyperparameter_tuning and spec.tune is not None:
        return spec.tune(
            X_fit, y_fit, cv_folds=tune_cv_folds,
            random_state=model_random_state, n_iter=tune_n_iter,
        )
    return None


def _score_fold_models(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    specs: Sequence[ModelSpec],
    *,
    kf: KFold,
    score_inner: bool,
    hyperparameter_tuning: bool,
    best_params: Optional[Mapping[str, Dict]],
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

    test_scores = _empty_scores(specs)
    inner_scores = _empty_scores(specs) if score_inner else None
    inner_sizes: List[int] = []

    for fit_idx, inner_idx in kf.split(X_train_full):
        X_fit, y_fit = X_train_full.iloc[fit_idx], y_train_full.iloc[fit_idx]
        X_inner, y_inner = X_train_full.iloc[inner_idx], y_train_full.iloc[inner_idx]
        inner_sizes.append(len(inner_idx))

        for spec in specs:
            params = _resolve_params(
                spec, X_fit, y_fit, best_params, hyperparameter_tuning,
                model_random_state, tune_cv_folds, tune_n_iter,
            )
            model = spec.train(X_fit, y_fit, params=params, random_state=model_random_state)

            for metric, value in compute_metrics(y_test, model.predict(X_test)).items():
                test_scores[spec.name][metric].append(value)
            if score_inner:
                for metric, value in compute_metrics(y_inner, model.predict(X_inner)).items():
                    inner_scores[spec.name][metric].append(value)

    return (
        _FoldScores(test_scores, len(test_idx)),
        _FoldScores(inner_scores, int(round(float(np.mean(inner_sizes))))) if score_inner else None,
    )


def _metric_rows(scenario, split_id, seed, split_type, scores: _FoldScores) -> List[Dict]:
    """Table 2 rows: numeric mean/std per model, formatted only at print time.

    The std is across inner folds, which all score the *same* fixed test set, so
    it measures sensitivity to the training subsample — not test-set
    uncertainty, and not something two models can be significance-tested on.
    """
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
    splits: Sequence[Split],
    specs: Sequence[ModelSpec],
    *,
    scenario: str,
    seed: int,
    cv_folds: int,
    shuffle: bool,
    hyperparameter_tuning: bool,
    best_params: Optional[Mapping[str, Dict]] = None,
    model_random_state: int = 0,
    tune_cv_folds: int = DEFAULT_TUNE_CV_FOLDS,
    tune_n_iter: int = DEFAULT_TUNE_N_ITER,
    controls: Optional[Mapping[str, Split]] = None,
    on_skip=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score each split's fixed OOD test set plus its in-distribution references.

    model_random_state seeds model construction and hyperparameter search
    (independent of `seed`, which seeds the inner KFold split on TRAIN).

    tune_cv_folds/tune_n_iter bound the hyperparameter search, which re-runs per
    fold per split per seed — the most expensive place in the codebase to leave
    the budget unbounded.

    Args:
        X: Feature matrix for the whole pool.
        y: Target for the whole pool.
        splits: The OOD splits to score.
        specs: Resolved model specifications.
        scenario: Scenario name, carried into the output tables.
        seed: Seed for the inner KFold split on the training portion.
        cv_folds: Inner folds per split.
        shuffle: Shuffle before the inner split.
        hyperparameter_tuning: Search inside every inner fold.
        best_params: Fixed parameters per model key, bypassing the search.
        model_random_state: Seed for model construction and the search.
        tune_cv_folds: Inner folds for the search.
        tune_n_iter: Candidates sampled by a randomized search.
        controls: Size-matched random split per split_id, or None to skip the
            ID-random reference. Built by the caller — this module scores the
            splits it is given rather than deciding which exist.
        on_skip: Optional `(split_id, reason) -> None` callback for splits that
            are too small to score, so the caller can report them.

    Returns:
        (table1, table2) — split summary, and metrics by model and split_type.
        Model-vs-model significance is deliberately not computed here; see
        summarize_model_comparison.
    """
    summary_rows, metrics_rows = [], []

    for split_id, train_idx, test_idx in splits:
        if len(train_idx) < cv_folds:
            if on_skip is not None:
                on_skip(split_id, f"n_train={len(train_idx)} < cv_folds={cv_folds}")
            continue

        kf = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=seed if shuffle else None)
        score = partial(
            _score_fold_models,
            X, y, specs=specs, kf=kf,
            hyperparameter_tuning=hyperparameter_tuning, best_params=best_params,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds, tune_n_iter=tune_n_iter,
        )

        ood_scores, id_paired_scores = score(train_idx, test_idx, score_inner=True)

        summary_rows.append(dict(
            scenario=scenario, split_id=split_id,
            heldout_target=heldout_target(split_id),
            n_train=len(train_idx), n_test=len(test_idx), seed=seed,
        ))

        metrics_rows += _metric_rows(scenario, split_id, seed, OOD, ood_scores)
        metrics_rows += _metric_rows(scenario, split_id, seed, ID_PAIRED, id_paired_scores)

        control = (controls or {}).get(split_id)
        if control is not None:
            _, control_train_idx, control_test_idx = control
            control_scores, _ = score(control_train_idx, control_test_idx, score_inner=False)
            metrics_rows += _metric_rows(scenario, split_id, seed, ID_RANDOM, control_scores)

    return pd.DataFrame(summary_rows), pd.DataFrame(metrics_rows)


def summarize_model_comparison(
    metrics_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metrics: Sequence[str] = ("mse", "mae"),
    split_type: str = OOD,
    min_pairs: int = MIN_PAIRS_FOR_TEST,
) -> pd.DataFrame:
    """Paired comparison of two models, one observation per OOD split.

    This replaces a per-split test over inner folds. Those folds all scored the
    *same* fixed test set, so their scores were repeated measurements of one
    quantity rather than independent observations of a difference — pairing them
    inflated the apparent evidence, and with a handful of folds the reported
    p-value was pinned near the test's own floor anyway.

    Pairing across splits is the valid version: each split is a different test
    set, and the two models saw identical training data on it.

    Args:
        metrics_df: Table 2, concatenated across splits and seeds.
        model_a, model_b: Model names to compare.
        metrics: Which metrics to compare; lower-is-better metrics only.
        split_type: Which rows to compare on.
        min_pairs: Below this many splits, p-values are withheld with a note.

    Returns:
        One row per (scenario, metric), or an empty frame if the requested
        models or split_type are absent.
    """
    if metrics_df.empty or "split_type" not in metrics_df.columns:
        return pd.DataFrame()

    subset = metrics_df[metrics_df["split_type"] == split_type]
    if subset.empty or not {model_a, model_b}.issubset(set(subset["model"])):
        return pd.DataFrame()

    rows = []
    for scenario, group in subset.groupby("scenario", sort=True):
        # One paired observation per split, averaging over seeds first. Seeds
        # differ only in the inner KFold and score the *same* test set, so
        # treating them as separate observations would reintroduce a milder
        # version of the non-independence this function exists to avoid.
        wide = group.pivot_table(
            index="split_id", columns="model",
            values=[f"{m.upper()}_mean" for m in metrics], aggfunc="mean",
        )
        for metric in metrics:
            column = f"{metric.upper()}_mean"
            if (column, model_a) not in wide.columns or (column, model_b) not in wide.columns:
                continue
            pair = wide[[(column, model_a), (column, model_b)]].dropna()
            if pair.empty:
                continue

            paired = {
                model_a: {metric: pair[(column, model_a)].tolist()},
                model_b: {metric: pair[(column, model_b)].tolist()},
            }
            result = compare_models_significance(
                paired, model_a, model_b, metric=metric, min_pairs=min_pairs
            )
            rows.append(dict(
                scenario=scenario, metric=metric.upper(),
                model_a=model_a, model_b=model_b,
                n_splits=result.n_pairs,
                mean_difference=result.mean_difference,
                t_pvalue=result.t_pvalue,
                wilcoxon_pvalue=result.w_pvalue,
                significant=bool(
                    result.note is None
                    and ((result.t_pvalue < 0.05) or (result.w_pvalue < 0.05))
                ),
                note=result.note or "",
            ))

    return pd.DataFrame(rows)


def summarize_runs_across_splits(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Average Table 2's per-split means into one row per (scenario, split_type, model).

    The reported spread is the std *of the per-split means*, i.e. how much a
    model's score moves between OOD splits — not the within-split fold spread,
    which stays in Table 2. n_splits is how many rows the summary covers; a NaN
    metric is dropped from that metric's mean only, and stays visible in Table 2
    rather than propagating into everything.

    Args:
        metrics_df: Table 2, concatenated across splits and seeds.
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
