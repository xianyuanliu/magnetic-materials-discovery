"""Applying a fitted model to new compositions: what is this material's predicted property.

The evaluation modes answer "how good is this model"; this one answers "what does it say about a material nobody has
made yet". It is also where a fitted model leaves the process — `fit_bundle` pairs the estimator with the feature
contract a later run needs to reuse it, which is what makes a saved model more than a pickle.

Reads no file and takes no path: the caller supplies the training rows, the compositions and the element reference
data, and decides where a bundle is loaded from or saved to.
"""

import warnings
from typing import List, Tuple

import pandas as pd

from prepdata.alloy_descriptors import add_engineered_features
from utils.persistence import TRAINING_FINGERPRINT, ModelBundle, align_features, fingerprint_training_data
from utils.registry import ModelSpec

# Column the predicted value is written to.
PREDICTION_COLUMN = "predicted"


def fit_bundle(
    X: pd.DataFrame,
    y: pd.Series,
    spec: ModelSpec,
    *,
    target_column: str,
    dataset: str,
    model_random_state: int = 0,
) -> ModelBundle:
    """Fit `spec` on every row given and wrap it with the contract needed to apply it later.

    Every row on purpose, unlike the evaluation modes: this model is going to be used, not scored, so holding rows
    back would only make it worse at the job it exists for.

    Args:
        X: Feature matrix; its column order becomes the bundle's feature contract.
        y: Target.
        spec: The model specification to fit.
        target_column: Name of the property being predicted, carried for the output table's benefit.
        dataset: Name of the dataset fitted on, carried so a loaded bundle can say where it came from.
        model_random_state: Seed for model construction.

    Returns:
        The fitted model with its feature columns, target, provenance and training-set size.
    """
    model = spec.train(X, y, hyperparams=None, random_state=model_random_state)
    return ModelBundle(
        model=model,
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


def check_bundle_matches(bundle: ModelBundle, X: pd.DataFrame, y: pd.Series) -> None:
    """Check that `bundle` was fitted on exactly these rows, and refuse it if not.

    A saved model outlives the dataset it came from. When the dataset is then corrected — rows deduplicated, a
    threshold changed, a feature rebuilt — the stale model keeps loading and keeps returning numbers that look
    entirely reasonable. This is the only place that difference is visible, so it raises rather than warns.

    Args:
        bundle: The loaded bundle.
        X: The feature matrix built from the dataset the bundle is about to be used alongside.
        y: Its target.

    Raises:
        ValueError: If the bundle was fitted on different rows, different features, or a different feature order.
    """
    recorded = bundle.metadata.get(TRAINING_FINGERPRINT)
    if recorded is None:
        warnings.warn(
            f"This bundle predates training fingerprints, so it cannot be checked against the current dataset. "
            f"It reports {bundle.describe()}. Refit it with predict_retrain: true to record one."
        )
        return

    current = fingerprint_training_data(X, y)
    if recorded != current:
        raise ValueError(
            f"The saved model was fitted on different data than the dataset now configured: it recorded "
            f"{recorded} over {bundle.n_train} row(s), the current table fingerprints as {current} over {len(X)} "
            f"row(s). Predictions from it would look reasonable and be wrong. Set predict_retrain: true to refit, "
            f"or point predict_model_path at the bundle that matches."
        )


def predict_formulas(
    bundle: ModelBundle,
    formulas: pd.DataFrame,
    pt: pd.DataFrame,
    mm: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> Tuple[pd.DataFrame, List[str]]:
    """Featurize `formulas` and predict the bundle's target for each.

    Args:
        bundle: A fitted model with its feature contract.
        formulas: Frame with a formula column.
        pt: Periodic table data, indexed by element symbol.
        mm: Symmetrized Miedema mixing-enthalpy matrix.
        formula_column: Name of the formula column.

    Returns:
        (predictions, skipped), where `predictions` has the formula and the predicted value, and `skipped` lists
        formulas that could not be featurized — an unparseable formula, or one whose elements are absent from the
        periodic table file, yields NaN features rather than a plausible-looking zero vector, so it is reported
        instead of predicted.
    """
    featurized = add_engineered_features(formulas, pt, mm, formula_column=formula_column)
    features = align_features(featurized, bundle.feature_columns, source="the featurized input")

    usable = features.notna().all(axis=1)
    skipped = featurized.loc[~usable, formula_column].astype(str).tolist()

    predictions = pd.DataFrame({
        formula_column: featurized.loc[usable, formula_column].to_numpy(),
        PREDICTION_COLUMN: bundle.model.predict(features[usable]),
    })
    return predictions, skipped
