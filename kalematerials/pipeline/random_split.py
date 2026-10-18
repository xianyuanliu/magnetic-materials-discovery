"""Repeated random train/validation splits."""

from typing import Dict, NamedTuple, Sequence

import pandas as pd

from kalematerials.evaluate.formatting import format_split_results, format_summary
from kalematerials.evaluate.metrics import build_result_rows, compute_metrics, summarize_scores
from kalematerials.loaddata.splits import build_random_split, ID_RANDOM, split_sizes
from kalematerials.predict.fitting import fit_models
from kalematerials.utils.registry import ModelSpec
from kalematerials.utils.reporting import Logger, SILENT


class RandomSplitResult(NamedTuple):
    """Result rows, their summary, the split sizes, and the first seed's models keyed by registry key."""

    results: pd.DataFrame
    summary: pd.DataFrame
    splits: pd.DataFrame
    first_split_models: Dict[str, object]


def evaluate_random_splits(
    X: pd.DataFrame,
    y: pd.Series,
    specs: Sequence[ModelSpec],
    *,
    seeds: Sequence[int] = (0,),
    train_size: float = 0.8,
    hyperparameter_tuning: bool = False,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
    model_random_state: int = 0,
    logger: Logger = SILENT,
) -> RandomSplitResult:
    """Evaluate a random train/validation split per seed.

    `seeds` control splits; `model_random_state` controls models and tuning. Tuning uses only each
    split's training rows, with search settings from `fit_models`.
    """
    collected, sizes = [], []
    first_split_models: Dict[str, object] = {}

    for run_i, seed in enumerate(seeds, start=1):
        logger.write(f"\n=== Random Split Run {run_i}/{len(seeds)} (split seed={seed}) ===\n")

        split = build_random_split(len(X), train_size, int(seed), split_id=f"seed{seed}")
        split_id, train_idx, valid_idx = split
        sizes.append(split_sizes([split], scenario="random_split", seed=int(seed)))
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        trained = fit_models(
            X_train,
            y_train,
            specs,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )
        predictions = {spec.name: trained[spec.key].predict(X_valid) for spec in specs}

        logger.write(format_split_results(y_valid, predictions))

        per_seed = {name: compute_metrics(y_valid, y_pred) for name, y_pred in predictions.items()}
        collected.append(
            build_result_rows(
                {name: {metric: [value] for metric, value in metrics.items()} for name, metrics in per_seed.items()},
                scenario="random_split",
                split_type=ID_RANDOM,
                split_ids=[split_id],
                seed=int(seed),
            )
        )

        if not first_split_models:
            first_split_models = trained

    results = pd.concat(collected, ignore_index=True)
    summary = summarize_scores(results)
    if len(seeds) > 1:
        logger.write(format_summary(summary, f"Random-Split Metrics across {len(seeds)} seed(s) (mean ± std):"))
    return RandomSplitResult(
        results=results,
        summary=summary,
        splits=pd.concat(sizes, ignore_index=True),
        first_split_models=first_split_models,
    )
