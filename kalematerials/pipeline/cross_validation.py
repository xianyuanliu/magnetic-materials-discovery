"""Repeated K-fold evaluation."""

from typing import NamedTuple, Sequence

import pandas as pd

from kalematerials.evaluate.formatting import format_cross_validation_results, format_summary
from kalematerials.evaluate.metrics import compute_metrics, FoldScores, METRICS, scores_to_results, summarize_scores
from kalematerials.loaddata.splits import build_kfold_splits, Split, split_sizes
from kalematerials.predict.fitting import fit_models
from kalematerials.utils.registry import ModelSpec
from kalematerials.utils.reporting import format_search_budget, Logger, SILENT


class CrossValidationResult(NamedTuple):
    """Long-form result rows, their mean/std summary and the split sizes (loaddata.splits.SPLIT_COLUMNS)."""

    results: pd.DataFrame
    summary: pd.DataFrame
    splits: pd.DataFrame


def score_models_on_splits(
    X: pd.DataFrame,
    y: pd.Series,
    specs: Sequence[ModelSpec],
    splits: Sequence[Split],
    hyperparameter_tuning: bool = False,
    model_random_state: int = 0,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
) -> FoldScores:
    """Fit models on each training split and score the validation rows.

    `splits` contains (split_id, train_idx, valid_idx) tuples. Return {model: {metric: [scores]}}
    in split order. Tuning uses only training rows; search settings follow `fit_models`.
    """
    results: FoldScores = {spec.name: {metric: [] for metric in METRICS} for spec in specs}

    for _, train_idx, valid_idx in splits:
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        fitted = fit_models(
            X_train,
            y_train,
            specs,
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


def cross_validate(
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
    logger: Logger = SILENT,
) -> CrossValidationResult:
    """Run K-fold cross-validation once per seed and return result rows, their summary and split sizes.

    `seeds` control fold shuffling; `model_random_state` controls models and tuning. Tuning uses only
    each fold's training rows, with search settings from `fit_models`.
    """
    if hyperparameter_tuning:
        logger.write(format_search_budget(len(seeds), folds, sum(1 for spec in specs if spec.tune is not None)))

    collected, sizes = [], []
    for run_i, seed in enumerate(seeds, start=1):
        logger.write(f"\n=== CV Run {run_i}/{len(seeds)} (seed={seed}) ===\n")

        splits = build_kfold_splits(len(X), folds, shuffle, int(seed))
        results = score_models_on_splits(
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
        sizes.append(split_sizes(splits, scenario="cross_validation", seed=int(seed)))

        logger.write(format_cross_validation_results(results))

    results = pd.concat(collected, ignore_index=True)
    summary = summarize_scores(results)
    if len(seeds) > 1:
        logger.write(format_summary(summary, f"Cross-Validation Metrics across {len(seeds)} seed(s) (mean ± std):"))
    return CrossValidationResult(results=results, summary=summary, splits=pd.concat(sizes, ignore_index=True))
