"""Repeated K-fold evaluation, as a function of data and numbers rather than of a run config.

Takes a feature matrix, a target and resolved model specifications; returns the long-form result rows. It reads no
file, resolves no path and prints nothing on its own, so a notebook or another project can call it with three
arguments and read the frame that comes back.

Progress is written to a `Logger` the caller supplies, defaulting to one that discards it. Finished numbers come back
as the return value; the caller decides whether to print or save them.
"""

from typing import Optional, Sequence, Tuple

import pandas as pd

from evaluate.metrics import METRICS, FoldScores, compute_metrics, scores_to_results
from loaddata.splits import Split, build_kfold_splits
from predict.fitting import fit_models
from utils.registry import ModelSpec
from utils.reporting import SILENT, Logger, format_comparisons, format_cross_validation_results, format_search_budget


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    specs: Sequence[ModelSpec],
    splits: Sequence[Split],
    hyperparameter_tuning: bool = False,
    model_random_state: int = 0,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
) -> FoldScores:
    """Score the given models on splits the caller has already built.

    Which rows form each fold is not decided here: the caller passes splits from loaddata/splits.py, so a plain K-fold
    and a grouped or shifted split go through this same scoring path.

    When hyperparameter_tuning is set, the search re-runs inside every split (proper nested CV — a split's validation
    rows never inform its own search). That is why the search budget is a separate, smaller pair of knobs: tune_cv_folds
    inner folds and tune_n_iter sampled candidates, whose cost is multiplied by len(splits) x len(specs).

    Args:
        X: Feature matrix.
        y: Target.
        specs: Resolved model specifications to score.
        splits: (split_id, train_idx, valid_idx) tuples; the identifier is unused, scores are returned per split in
            the order given.
        hyperparameter_tuning: Search inside every split.
        model_random_state: Seed for model construction and the search.
        tune_cv_folds: Inner folds for the search.
        tune_n_iter: Candidates sampled by a randomized search.

    Returns:
        {model name: {metric: [one score per split]}}.
    """
    results: FoldScores = {
        spec.name: {metric: [] for metric in METRICS} for spec in specs
    }

    for _, train_idx, valid_idx in splits:
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        fitted = fit_models(
            X_train, y_train, specs,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )

        for spec in specs:
            metrics = compute_metrics(y_valid, fitted[spec.key].predict(X_valid))
            for metric in METRICS:
                results[spec.name][metric].append(metrics[metric])

    return results


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    specs: Sequence[ModelSpec],
    *,
    folds: int = 5,
    shuffle: bool = True,
    seeds: Sequence[int] = (0,),
    hyperparameter_tuning: bool = False,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
    model_random_state: int = 0,
    compare: Optional[Tuple[str, str]] = None,
    logger: Logger = SILENT,
) -> pd.DataFrame:
    """Score every model with K-fold, repeated once per seed.

    Each seed reshuffles the folds, so the spread across seeds measures how much the partition itself moved the
    result — which on a few-hundred-row dataset is often larger than the gap between two models.

    Args:
        X: Feature matrix.
        y: Target.
        specs: Resolved model specifications to score.
        folds: Folds per repeat.
        shuffle: Shuffle before partitioning; required for the seed to have any effect.
        seeds: One repeat per seed, seeding the partition only.
        hyperparameter_tuning: Search inside every fold, so a fold's validation rows never inform its own search.
        tune_cv_folds: Inner folds for that search.
        tune_n_iter: Candidates sampled per search.
        model_random_state: Seed for model construction and the search, kept separate from the split seeds.
        compare: Two model *names* to significance-test against each other after every repeat, or None to skip.
        logger: Where progress goes while the run is still going.

    Returns:
        The long-form result rows — one per (split, model, metric) — for every seed, in seed order.
    """
    if hyperparameter_tuning:
        logger.write(format_search_budget(
            len(seeds), folds, sum(1 for spec in specs if spec.tune is not None), tune_n_iter, tune_cv_folds,
        ))

    collected = []
    for run_i, seed in enumerate(seeds, start=1):
        logger.write(f"\n=== CV Run {run_i}/{len(seeds)} (seed={seed}) ===\n")

        splits = build_kfold_splits(len(X), folds, shuffle, int(seed))
        results = cross_validate_models(
            X,
            y,
            specs,
            splits,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )
        collected.append(scores_to_results(results, splits, scenario="cross_validation", seed=int(seed)))

        logger.write(format_cross_validation_results(results))
        if compare is not None:
            # Every fold trains on all but one of K parts, so one fold's test set is 1 / (K - 1) of its
            # training set. The folds share training data; compare_models_significance corrects for it.
            logger.write(format_comparisons(results, *compare, test_train_ratio=1.0 / (folds - 1)))

    return pd.concat(collected, ignore_index=True)
