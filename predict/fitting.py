"""Fit a set of model specifications on one set of rows.

The one place a model gets fitted, so every evaluation mode resolves hyperparameters identically.
"""

from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

from utils.registry import ModelSpec


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    specs: Sequence[ModelSpec],
    *,
    hyperparameter_tuning: bool = False,
    best_hyperparams: Optional[Mapping[str, Dict]] = None,
    model_random_state: int = 0,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
) -> Dict[str, object]:
    """Fit one model per spec on the given rows, searching hyperparameters inside those same rows when asked.

    The one place a model gets fitted, so every evaluation mode resolves hyperparameters identically: a fixed set from
    `best_hyperparams` wins, otherwise a search runs when `hyperparameter_tuning` is on and the spec has one, otherwise
    the spec's own defaults apply. Searching inside the training rows is what keeps a split's held-out rows out of its
    own search.

    Callers that only want scores go through `cross_validate_models`. This is for the ones that also need the fitted
    estimator afterwards — to explain it, to read its ensemble spread, or to score it on a second set.

    Args:
        X_train, y_train: The rows to fit on.
        specs: Resolved model specifications.
        hyperparameter_tuning: Search inside these rows.
        best_hyperparams: Fixed parameters per model key, bypassing the search.
        model_random_state: Seed for model construction and the search.
        tune_cv_folds: Inner folds for the search.
        tune_n_iter: Candidates sampled by a randomized search.

    Returns:
        {spec.key: fitted estimator}, in the order the specs were given.
    """
    fitted: Dict[str, object] = {}
    for spec in specs:
        hyperparams = None
        if best_hyperparams is not None and spec.key in best_hyperparams:
            hyperparams = best_hyperparams[spec.key]
        elif hyperparameter_tuning and spec.tune is not None:
            hyperparams = spec.tune(
                X_train, y_train, cv_folds=tune_cv_folds, random_state=model_random_state, n_iter=tune_n_iter,
            )
        fitted[spec.key] = spec.train(X_train, y_train, hyperparams=hyperparams, random_state=model_random_state)
    return fitted
