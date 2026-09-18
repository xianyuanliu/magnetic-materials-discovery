"""Config adapter for `evaluation_mode: predict`.

Two ways in:
  - No saved bundle (or `predict_retrain: true`) — fit on the whole dataset and save the bundle to
    `predict_model_path`.
  - A saved bundle — load it and skip training entirely.

Either way the compositions to score come from `predict_input_path` (a CSV) or `predict_formulas` (a list in the
config), are featurized with exactly the same code as the training data, and are written out with their predictions.
"""

import warnings
from pathlib import Path
from typing import Mapping

import pandas as pd

from config import RunConfig
from loaddata.tabular_access import load_element_properties, load_feature_table
from pipeline.inference import fit_bundle, predict_formulas
from utils.persistence import ModelBundle, load_model_bundle, save_model_bundle, save_tables
from utils.registry import ModelSpec, resolve_models
from utils.reporting import print_predictions


def _read_formulas(cfg: RunConfig) -> pd.DataFrame:
    """Collect the compositions to predict, from a CSV or from the config.

    Returns:
        A single-column frame of formulas.

    Raises:
        ValueError: If neither source is configured, if the CSV lacks the formula column, or if no usable formula is
            left.
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


def _resolve_bundle(cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> ModelBundle:
    """Load the saved bundle if there is one to load, otherwise fit and save."""
    path = cfg.predict.model_path
    if path and Path(path).exists() and not cfg.predict.retrain:
        bundle = load_model_bundle(path)
        print(f"Loaded model from {Path(path).resolve()}")
        return bundle

    (spec,) = resolve_models(registry, [cfg.predict.model])
    X, y, _ = load_feature_table(
        cfg.dataset_path,
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )
    print(f"Fitting {spec.name} on all {len(X)} row(s) of {cfg.dataset_path}")
    bundle = fit_bundle(
        X, y, spec,
        target_column=cfg.target_column,
        dataset=cfg.dataset,
        model_random_state=cfg.model_random_state,
    )

    if path:
        print(f"Saved model to {save_model_bundle(path, bundle)}")
    else:
        print("[INFO] No 'predict_model_path' set, so this model is not saved and " "the next run will refit it.")
    return bundle


def run(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> pd.DataFrame:
    """Resolve a model, score the configured compositions, and save the output.

    Args:
        cfg: The resolved run configuration.
        registry: Registry to resolve `predict_model` against.

    Returns:
        The prediction table, one row per successfully featurized composition.
    """
    bundle = _resolve_bundle(cfg, registry)
    print(f"Model: {bundle.describe()}")

    formulas = _read_formulas(cfg)
    pt, mm = load_element_properties(cfg.pt_path, cfg.mm_path)
    predictions, skipped = predict_formulas(bundle, formulas, pt, mm, formula_column=cfg.formula_column)

    if skipped:
        warnings.warn(f"Could not featurize {len(skipped)} formula(s): {skipped}")

    print(f"\nPredicted {bundle.target_column} for {len(predictions)} composition(s):")
    if cfg.print_results:
        print_predictions(predictions)

    if cfg.predict.output_path and cfg.save_results:
        destination = Path(cfg.predict.output_path)
        save_tables({destination.name: predictions}, str(destination.parent))
        print(f"\nSaved predictions to: {destination.resolve()}")

    return predictions
