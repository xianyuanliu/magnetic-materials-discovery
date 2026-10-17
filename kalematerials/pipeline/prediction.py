"""Fit model bundles and predict new compositions."""

import warnings
from typing import List, Sequence, Tuple

import pandas as pd

from kalematerials.embed.composition_descriptors import add_engineered_features
from kalematerials.evaluate.metrics import best_model
from kalematerials.predict.fitting import fit_models
from kalematerials.utils.persistence import align_features, fingerprint_training_data, ModelBundle, TRAINING_FINGERPRINT
from kalematerials.utils.registry import ModelSpec

PREDICTION_COLUMN = "predicted"


def fit_bundle(
    X: pd.DataFrame,
    y: pd.Series,
    spec: ModelSpec,
    *,
    target_column: str,
    dataset: str,
    hyperparameter_tuning: bool = False,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
    model_random_state: int = 0,
) -> ModelBundle:
    """Fit all rows and bundle the model with feature order, target and training-data fingerprint.

    Search and seed settings follow `fit_models`.
    """
    fitted = fit_models(
        X,
        y,
        [spec],
        hyperparameter_tuning=hyperparameter_tuning,
        model_random_state=model_random_state,
        tune_cv_folds=tune_cv_folds,
        tune_n_iter=tune_n_iter,
    )
    return ModelBundle(
        model=fitted[spec.key],
        model_key=spec.key,
        model_name=spec.name,
        feature_columns=tuple(X.columns),
        target_column=target_column,
        dataset=dataset,
        n_train=len(X),
        metadata={
            "model_random_state": model_random_state,
            TRAINING_FINGERPRINT: fingerprint_training_data(X, y),
        },
    )


def fit_best_bundle(
    X: pd.DataFrame,
    y: pd.Series,
    specs: Sequence[ModelSpec],
    summary: pd.DataFrame,
    *,
    target_column: str,
    dataset: str,
    metric: str = "mse",
    hyperparameter_tuning: bool = False,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
    model_random_state: int = 0,
) -> ModelBundle:
    """Refit the spec with the best mean `metric` in `summary` on all rows and bundle it, as in `fit_bundle`."""
    winner = best_model(summary, metric=metric)
    (spec,) = [spec for spec in specs if spec.name == winner]
    return fit_bundle(
        X,
        y,
        spec,
        target_column=target_column,
        dataset=dataset,
        hyperparameter_tuning=hyperparameter_tuning,
        tune_cv_folds=tune_cv_folds,
        tune_n_iter=tune_n_iter,
        model_random_state=model_random_state,
    )


def check_bundle_matches(bundle: ModelBundle, X: pd.DataFrame, y: pd.Series) -> None:
    """Raise ValueError if the bundle's training-data fingerprint differs from (X, y)."""
    recorded = bundle.metadata.get(TRAINING_FINGERPRINT)
    if recorded is None:
        warnings.warn(f"Bundle has no training fingerprint, so it cannot be checked: {bundle.describe()}")
        return

    current = fingerprint_training_data(X, y)
    if recorded != current:
        raise ValueError(
            f"Bundle was fitted on different data: fingerprint {recorded} over {bundle.n_train} row(s), "
            f"current data {current} over {len(X)} row(s). Refit it or load the matching bundle."
        )


def predict_formulas(
    bundle: ModelBundle,
    formulas: pd.DataFrame,
    periodic_table: pd.DataFrame,
    miedema: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> Tuple[pd.DataFrame, List[str]]:
    """Featurize formulas and return (predictions, skipped formulas).

    `predictions` contains the formula and PREDICTION_COLUMN. `periodic_table` is indexed by element symbol;
    `miedema` is a symmetric element-pair matrix. Formulas with incomplete features are skipped.
    """
    featurized = add_engineered_features(formulas, periodic_table, miedema, formula_column=formula_column)
    features = align_features(featurized, bundle.feature_columns, source="the featurized input")

    usable = features.notna().all(axis=1)
    skipped = featurized.loc[~usable, formula_column].astype(str).tolist()

    predictions = pd.DataFrame(
        {
            formula_column: featurized.loc[usable, formula_column].to_numpy(),
            PREDICTION_COLUMN: bundle.model.predict(features[usable]),
        }
    )
    return predictions, skipped
