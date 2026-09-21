"""Repeated random train/valid splits, as a function of data and numbers rather than of a run config.

One split per seed, every model fitted on the same training rows and scored on the same held-out rows, so the spread
across seeds measures how much the draw itself moved the result. Reads no file, resolves no path, prints nothing on
its own.

The models fitted on the first seed's split come back with the results, because explaining a model means holding the
fitted estimator — and refitting it to explain it would be a second, differently seeded model.
"""

from typing import Dict, Sequence, Tuple

import pandas as pd

from predict.fitting import fit_models
from evaluate.metrics import build_result_rows, compute_metrics
from loaddata.splits import build_random_split
from utils.registry import ModelSpec
from utils.reporting import SILENT, Logger, format_split_results


def run_random_split(
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
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Score every model on a random train/valid split, repeated once per seed.

    Args:
        X: Feature matrix.
        y: Target.
        specs: Resolved model specifications to score.
        seeds: One repeat per seed, seeding the split only.
        train_size: Fraction of rows used for training in each split.
        hyperparameter_tuning: Search inside each split's training rows, so its held-out rows never inform the search.
        tune_cv_folds: Inner folds for that search.
        tune_n_iter: Candidates sampled per search.
        model_random_state: Seed for model construction and the search, kept separate from the split seeds.
        logger: Where progress goes while the run is still going.

    Returns:
        (results, models): the long-form result rows for every seed, and the models fitted on the first seed's split,
        keyed by model key.
    """
    collected = []
    first_split_models: Dict[str, object] = {}

    for run_i, seed in enumerate(seeds, start=1):
        logger.write(f"\n=== Random Split Run {run_i}/{len(seeds)} (split seed={seed}) ===\n")

        _, train_idx, valid_idx = build_random_split(len(X), train_size, int(seed))
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        trained = fit_models(
            X_train, y_train, specs,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )
        predictions = {spec.name: trained[spec.key].predict(X_valid) for spec in specs}

        logger.write(format_split_results(y_valid, predictions))

        per_seed = {name: compute_metrics(y_valid, y_pred) for name, y_pred in predictions.items()}
        collected.append(build_result_rows(
            {name: {metric: [value] for metric, value in metrics.items()} for name, metrics in per_seed.items()},
            scenario="random_split", split_type="ID", split_ids=[f"seed{seed}"], seed=int(seed),
        ))

        if not first_split_models:
            first_split_models = trained

    return pd.concat(collected, ignore_index=True), first_split_models
