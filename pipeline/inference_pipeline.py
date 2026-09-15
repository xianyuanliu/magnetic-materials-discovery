"""Property prediction: fit a model, save it, and apply it to new compositions.

The other pipelines answer "how good is this model"; this one answers "what is the predicted property of this material".
It is also where a fitted model leaves the process: without persistence every run retrained from scratch and discarded
its models, so nothing downstream could use them.

Two ways in:
  - No saved bundle (or `predict_retrain: true`) — fit on the whole dataset and
    save the bundle to `predict_model_path`.
  - A saved bundle — load it and skip training entirely.

Either way the compositions to score come from `predict_input_path` (a CSV) or `predict_formulas` (a list in the
config), are featurized with exactly the same code as the training data, and are written out with their predictions.
"""

from pathlib import Path
from typing import List, Mapping, Tuple

import pandas as pd

from config import RunConfig
from utils.registry import ModelSpec, resolve_models
from loaddata.feature_csv_access import load_features_and_target
from loaddata.raw_loaders import load_elemental_data
from utils.persistence import ModelBundle, align_features, load_model_bundle, save_model_bundle
from prepdata.build_features import add_engineered_features
from utils.reporting import print_predictions

# Column the predicted value is written to.
PREDICTION_COLUMN = "predicted"


def _read_formulas(cfg: RunConfig) -> pd.DataFrame:
    """Collect the compositions to predict, from a CSV or from the config.

    Returns:
        A single-column frame of formulas.

    Raises:
        ValueError: If neither source is configured, if the CSV lacks the
            formula column, or if no usable formula is left.
    """
    if cfg.predict.input_path:
        data = pd.read_csv(cfg.predict.input_path)
        if cfg.formula_column not in data.columns:
            raise ValueError(
                f"{cfg.predict.input_path} has no {cfg.formula_column!r} column. "
                f"Found: {sorted(data.columns)}"
            )
        formulas = data[[cfg.formula_column]].copy()
    elif cfg.predict.formulas:
        formulas = pd.DataFrame({cfg.formula_column: list(cfg.predict.formulas)})
    else:
        raise ValueError(
            "predict mode needs compositions: set 'predict_input_path' to a CSV "
            "or 'predict_formulas' to a list in the config."
        )

    formulas = formulas.dropna(subset=[cfg.formula_column]).reset_index(drop=True)
    if formulas.empty:
        raise ValueError("No usable chemical formulas to predict.")
    return formulas


def _train_bundle(cfg: RunConfig, spec: ModelSpec) -> ModelBundle:
    """Fit `spec` on the whole configured dataset and wrap it in a bundle."""
    X, y, feature_columns = load_features_and_target(
        cfg.dataset_path,
        target_column=cfg.target_column,
        feature_columns=cfg.feature_columns,
        formula_column=cfg.formula_column,
    )
    # Every row on purpose, unlike the evaluation modes: this model is going to be used, not scored.
    print(f"Fitting {spec.name} on all {len(X)} row(s) of {cfg.dataset_path}")
    model = spec.train(X, y, hyperparams=None, random_state=cfg.model_random_state)

    return ModelBundle(
        model=model,
        model_key=spec.key,
        model_name=spec.name,
        feature_columns=tuple(feature_columns),
        target_column=cfg.target_column,
        dataset=cfg.dataset,
        n_train=len(X),
        metadata={"model_random_state": cfg.model_random_state},
    )


def _resolve_bundle(cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> ModelBundle:
    """Load the saved bundle if there is one to load, otherwise fit and save."""
    path = cfg.predict.model_path
    if path and Path(path).exists() and not cfg.predict.retrain:
        bundle = load_model_bundle(path)
        print(f"Loaded model from {Path(path).resolve()}")
        return bundle

    (spec,) = resolve_models(registry, [cfg.predict.model])
    bundle = _train_bundle(cfg, spec)
    if path:
        print(f"Saved model to {save_model_bundle(path, bundle)}")
    else:
        print("[INFO] No 'predict_model_path' set, so this model is not saved and " "the next run will refit it.")
    return bundle


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
        mm: Symmetrised Miedema mixing-enthalpy matrix.
        formula_column: Name of the formula column.

    Returns:
        (predictions, skipped), where `predictions` has the formula and the
        predicted value, and `skipped` lists formulas that could not be
        featurized — an unparseable formula, or one whose elements are absent
        from the periodic table file, yields NaN features rather than a
        plausible-looking zero vector, so it is reported instead of predicted.
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


def run_predict(*, cfg: RunConfig, model_registry: Mapping[str, ModelSpec]) -> pd.DataFrame:
    """Run the predict pipeline: resolve a model, score compositions, save output.

    Called from main.py when evaluation_mode == 'predict'.

    Args:
        cfg: The resolved run configuration.
        model_registry: Registry to resolve `predict_model` against.

    Returns:
        The prediction table, one row per successfully featurized composition.
    """
    bundle = _resolve_bundle(cfg, model_registry)
    print(f"Model: {bundle.describe()}")

    formulas = _read_formulas(cfg)
    pt, mm = load_elemental_data(cfg.pt_path, cfg.mm_path)
    predictions, skipped = predict_formulas(bundle, formulas, pt, mm, formula_column=cfg.formula_column)

    if skipped:
        print(f"\n[WARN] Could not featurize {len(skipped)} formula(s): {skipped}")

    print(f"\nPredicted {bundle.target_column} for {len(predictions)} composition(s):")
    print_predictions(predictions)

    if cfg.predict.output_path:
        destination = Path(cfg.predict.output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(destination, index=False)
        print(f"\nSaved predictions to: {destination.resolve()}")

    return predictions
