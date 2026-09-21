"""The one module that reads a run config: it turns config values into the arguments `pipeline/` takes.

Everything it calls works without a YAML schema — the stage packages take explicit arguments, and so do the
`pipeline/` functions, which return their numbers rather than printing or saving them. What is left is here: reading
the files a config names, resolving model keys against the registry, choosing where output goes, and deciding what
gets printed.

Keeping it in one module is the point. "Which code depends on the config format" has a one-file answer, and adding a
mode means adding a function here rather than another file whose name shadows one in `pipeline/`.

Each `run_*` here corresponds to one `evaluation_mode`, plus `run_data_visualization` for the optional plots any mode
can enable.
"""

import warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import RunConfig
from evaluate.metrics import METRICS, results_to_scores, wide_table_to_results
from case_study_references import CASE_STUDIES
from interpret.alloy_series import build_series_panels
from interpret.reference_comparison import plot_predictions_against_measurements
from interpret.model_explanations import plot_model_explanations
from interpret.dataset_distributions import (
    count_compounds_by_radix,
    plot_target_distribution_by_element,
    plot_target_violin_by_element,
)
from loaddata.splits import build_random_split
from loaddata.tabular_access import load_element_properties, load_feature_table, load_periodic_table
from pipeline.cross_validation import run_cross_validation as cross_validate
from pipeline.ood import build_scenarios, resolve_split_elements, run_ood_evaluation
from pipeline.uncertainty import (
    UNCERTAINTY_METRICS,
    resolve_uncertainty_model,
    run_uncertainty_evaluation,
)
from pipeline.prediction import check_bundle_matches, fit_bundle, predict_formulas
from pipeline.random_split import run_random_split as evaluate_random_split
from utils.persistence import ModelBundle, load_model_bundle, save_model_bundle, save_results, save_tables
from utils.registry import ModelSpec, resolve_models, resolve_trained_models
from utils.reporting import (
    SILENT,
    Logger,
    format_comparisons,
    print_compound_counts,
    print_ood_tables,
    print_predictions,
    print_summary,
    print_uncertainty_report,
)


# The transition metals this dataset is organized around: the distribution plots draw one subset per element, and
# these four are the ones whose presence moves the saturation magnetization.
DISTRIBUTION_ELEMENTS = ("Fe", "Co", "Cr", "Mn")

# Bin edges for the distribution plot, covering the saturation magnetization range in tesla.
DISTRIBUTION_BINS = np.arange(0.0, 2.6, 0.2)


def _load_features(cfg: RunConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix and target from the dataset the config names."""
    X, y, _ = load_feature_table(
        cfg.dataset_path,
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )
    return X, y


def _logger(cfg: RunConfig) -> Logger:
    """The sink a pipeline writes progress to: stdout when the run prints, discarded when it does not."""
    return Logger() if cfg.print_results else SILENT


def _compared_names(cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> Optional[Tuple[str, str]]:
    """Resolve `compare_models` into the two display names the significance tests use, or None."""
    if cfg.compare_models is None:
        return None
    name_a, name_b = (spec.name for spec in resolve_models(registry, cfg.compare_models))
    return name_a, name_b


def run_cross_validation(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Load what the config names, run repeated K-fold, then save and print what it asks for.

    Args:
        cfg: The resolved run configuration.
        registry: Registry to resolve `models` and `compare_models` against.
    """
    specs = resolve_models(registry, cfg.models)
    X, y = _load_features(cfg)

    # Resolved once: the pair is the same for every seed.
    compared = _compared_names(cfg, registry)

    results = cross_validate(
        X, y, specs,
        folds=cfg.kfold.folds,
        shuffle=cfg.kfold.shuffle,
        seeds=cfg.kfold.seeds,
        hyperparameter_tuning=cfg.tuning.enabled,
        tune_cv_folds=cfg.tuning.cv_folds,
        tune_n_iter=cfg.tuning.n_iter,
        model_random_state=cfg.model_random_state,
        compare=compared,
        logger=_logger(cfg),
    )

    summary, directory = save_results(
        results,
        cfg.results_output_dir,
        prefix=f"{cfg.prefix}_cv_",
        enabled=cfg.save_results,
    )
    # Across every fold of every seed, so cross-validation and random_split close on the same statement.
    if cfg.print_results and len(cfg.kfold.seeds) > 1:
        print_summary(summary, f"Cross-Validation Metrics across {len(cfg.kfold.seeds)} seed(s) (mean ± std):")
    if directory is not None and cfg.print_results:
        print(f"\nSaved cross-validation results to: {directory.resolve()}")


def _plot_interpretation(
    cfg: RunConfig,
    registry: Mapping[str, ModelSpec],
    trained: Mapping[str, object],
    feature_columns: Sequence[str],
    X_train,
    X_valid,
    y_valid,
) -> None:
    """Turn the figure settings into calls on `interpret/`.

    Config unpacking only: which fitted models the config names, where the figures go, and which reference files the
    case studies need. Both figures are config-free functions that take explicit arguments.
    """
    plots_dir = Path(cfg.plots_output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if cfg.interpret.explain_models:
        specs = resolve_trained_models(registry, trained, cfg.interpret.explain_models)
        plot_model_explanations(
            {spec.name: trained[spec.key] for spec in specs},
            X_train, X_valid, y_valid,
            save_dir=plots_dir, prefix=f"{cfg.prefix}_", random_state=cfg.model_random_state,
        )

    if cfg.interpret.case_study_models:
        specs = resolve_trained_models(registry, trained, cfg.interpret.case_study_models)
        pt, mm = load_element_properties(cfg.pt_path, cfg.mm_path)
        panels = build_series_panels(
            CASE_STUDIES,
            feature_columns,
            {spec.name: trained[spec.key] for spec in specs},
            pt,
            mm,
        )
        plot_predictions_against_measurements(
            panels,
            y_label=cfg.target_column,
            save_path=plots_dir / f"{cfg.prefix}_case_studies.png",
        )


def run_random_split(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Load what the config names, repeat the split once per seed, then save, print and explain what it asks for.

    `random_split_seeds` seeds the split only, kept separate from `random_state` (model construction and tuning).
    Explanation figures come from the first seed's models, so they stay put when a seed is added.

    Args:
        cfg: The resolved run configuration.
        registry: Registry to resolve `models`, `compare_models` and the interpretation model names against.
    """
    specs = resolve_models(registry, cfg.models)
    X, y = _load_features(cfg)

    results, first_split_models = evaluate_random_split(
        X, y, specs,
        seeds=cfg.random_split.seeds,
        train_size=cfg.random_split.train_size,
        hyperparameter_tuning=cfg.tuning.enabled,
        tune_cv_folds=cfg.tuning.cv_folds,
        tune_n_iter=cfg.tuning.n_iter,
        model_random_state=cfg.model_random_state,
        logger=_logger(cfg),
    )

    summary, directory = save_results(
        results,
        cfg.results_output_dir,
        prefix=f"{cfg.prefix}_random_split_",
        enabled=cfg.save_results,
    )
    if directory is not None and cfg.print_results:
        print(f"\nSaved random-split results to: {directory.resolve()}")

    if cfg.print_results and len(cfg.random_split.seeds) > 1:
        print_summary(summary, f"Random-Split Metrics across {len(cfg.random_split.seeds)} seed(s) (mean ± std):")
        compared = _compared_names(cfg, registry)
        if compared is not None:
            # One observation per seed, each a fresh split of the same rows, so the training sets overlap.
            print(format_comparisons(
                results_to_scores(results), *compared,
                test_train_ratio=(1.0 - cfg.random_split.train_size) / cfg.random_split.train_size,
            ), end="")

    if not cfg.interpret.enabled:
        return

    # The same split the models came from, rebuilt rather than carried: build_random_split is pure and seeded.
    _, train_idx, valid_idx = build_random_split(len(X), cfg.random_split.train_size, int(cfg.random_split.seeds[0]))
    _plot_interpretation(
        cfg, registry, first_split_models, list(X.columns),
        X.iloc[train_idx], X.iloc[valid_idx], y.iloc[valid_idx],
    )


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


def _load_training_table(cfg: RunConfig):
    """Build the feature table the config names, or None when it names no dataset at all."""
    if not cfg.dataset_path or not Path(cfg.dataset_path).exists():
        return None
    return _load_features(cfg)


def _resolve_bundle(cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> ModelBundle:
    """Load the saved bundle if there is one to load, otherwise fit and save.

    A loaded bundle is checked against the configured dataset when that dataset is present, because a model that
    outlived its training data keeps predicting and stops being right. A bundle shipped without the dataset is used
    as-is: there is nothing to check it against.
    """
    path = cfg.predict.model_path
    if path and Path(path).exists() and not cfg.predict.retrain:
        bundle = load_model_bundle(path)
        print(f"Loaded model from {Path(path).resolve()}")
        training = _load_training_table(cfg)
        if training is not None:
            check_bundle_matches(bundle, *training)
        return bundle

    (spec,) = resolve_models(registry, [cfg.predict.model])
    training = _load_training_table(cfg)
    if training is None:
        raise ValueError(
            f"Fitting a model needs training data, but dataset_path is {cfg.dataset_path!r} and no such file exists. "
            f"Point predict_model_path at a saved bundle to skip fitting, or set dataset_path."
        )

    X, y = training
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


def run_predict(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> pd.DataFrame:
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


def run_data_visualization(*, cfg: RunConfig) -> None:
    """Plot the target's distribution across the transition-metal subsets.

    Args:
        cfg: The resolved run configuration; `plots_output_dir` names where the figures go.
    """
    plots_dir = Path(cfg.plots_output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _, y, metadata = load_feature_table(
        cfg.dataset_path,
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )
    data = metadata.assign(**{cfg.target_column: y})

    plot_target_distribution_by_element(
        data,
        target_column=cfg.target_column,
        elements=DISTRIBUTION_ELEMENTS,
        formula_column=cfg.formula_column,
        bins=DISTRIBUTION_BINS,
        save_path=plots_dir / f"{cfg.prefix}_ms_distribution_by_tm.png",
    )
    plot_target_violin_by_element(
        data,
        target_column=cfg.target_column,
        elements=DISTRIBUTION_ELEMENTS,
        formula_column=cfg.formula_column,
        target_label="Saturation Magnetization (T)",
        title=f"{cfg.prefix.upper()} Violin Plot",
        save_path=plots_dir / f"{cfg.prefix}_violin_ms_by_tm.png",
    )
    print_compound_counts(count_compounds_by_radix(data, formula_column=cfg.formula_column))


# Result tables, in the order print_ood_tables takes them, paired with the file each is written to.
OOD_TABLE_FILENAMES = (
    "table1_splits_summary.csv",
    "table2_metrics_by_model.csv",
    "table3_model_comparison_significance.csv",
    "table4_combined_comparison.csv",
    "table5_generalization_gap.csv",
)


def _ood_scenarios(cfg: RunConfig, X, y, metadata):
    """Turn the config's OOD settings into the split families it selects."""
    elements_per_row, element_to_group, element_to_period = resolve_split_elements(
        metadata, cfg.formula_column, load_periodic_table(cfg.pt_path),
    )
    return build_scenarios(
        X, y, elements_per_row, element_to_group, element_to_period,
        mode=cfg.ood.mode,
        min_train=cfg.ood.min_train,
        min_test=cfg.ood.min_test,
        max_splits=cfg.ood.max_splits,
        elements=cfg.ood.elements,
        periods=cfg.ood.periods,
        groups=cfg.ood.groups,
        period_strict=cfg.ood.period_strict,
        group_strict=cfg.ood.group_strict,
        n_clusters=cfg.ood.k,
        cluster_seed=cfg.ood.cluster_seed,
        fractions=cfg.ood.fractions,
        sparsex_neighbors=cfg.ood.sparsex_neighbors,
        sparsey_center=cfg.ood.sparsey_center,
    )


def run_ood(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Build the split families the config selects, score them, then save and print the tables.

    Args:
        cfg: The resolved run configuration.
        registry: Registry to resolve `models` and `compare_models` against.
    """
    specs = resolve_models(registry, cfg.models)
    X, y, metadata = load_feature_table(
        cfg.dataset_path,
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )

    tables = run_ood_evaluation(
        X, y, specs, _ood_scenarios(cfg, X, y, metadata),
        seeds=cfg.ood.seeds,
        inner_folds=cfg.kfold.folds,
        inner_shuffle=cfg.kfold.shuffle,
        size_matched_control=cfg.ood.size_matched_control,
        hyperparameter_tuning=cfg.tuning.enabled,
        tune_cv_folds=cfg.tuning.cv_folds,
        tune_n_iter=cfg.tuning.n_iter,
        model_random_state=cfg.model_random_state,
        compare=_compared_names(cfg, registry),
        logger=_logger(cfg),
    )

    if cfg.print_results:
        print_ood_tables(*tables)

    out_dir = save_tables(dict(zip(OOD_TABLE_FILENAMES, tables)), cfg.ood.output_dir, enabled=cfg.save_results)
    if out_dir is not None and cfg.print_results:
        print(f"\nSaved OOD tables to: {out_dir.resolve()}")

    # The unified table reports one observation per split, matching table 2's *_mean columns: the inner folds all score
    # the same held-out rows, so their spread belongs in *_std rather than as separate observations.
    _, directory = save_results(
        wide_table_to_results(
            tables.metrics_by_model,
            {f"{metric.upper()}_mean": metric for metric in METRICS},
            identifiers={
                "scenario": "scenario", "split_type": "split_type", "split_id": "split_id",
                "seed": "seed", "model": "model",
            },
        ),
        cfg.ood.output_dir,
        enabled=cfg.save_results,
    )
    if directory is not None and cfg.print_results:
        print(f"Saved the unified result table to: {directory.resolve()}")


UNCERTAINTY_TABLE_FILENAMES = (
    "uncertainty_table1_by_split.csv",
    "uncertainty_table2_pooled_by_seed.csv",
    "uncertainty_table3_across_seeds.csv",
)


def run_uncertainty(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Score the configured model's intervals in distribution and under shift, then save and print the tables.

    Args:
        cfg: The resolved run configuration.
        registry: Registry to resolve `uncertainty_model` against.
    """
    spec = resolve_uncertainty_model(registry, cfg.uncertainty.model)
    X, y, metadata = load_feature_table(
        cfg.dataset_path,
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )

    tables = run_uncertainty_evaluation(
        X, y, spec, _ood_scenarios(cfg, X, y, metadata),
        seeds=cfg.uncertainty.seeds,
        alpha=cfg.uncertainty.alpha,
        calibration_fraction=cfg.uncertainty.calibration_fraction,
        inner_folds=cfg.kfold.folds,
        inner_shuffle=cfg.kfold.shuffle,
        size_matched_control=cfg.ood.size_matched_control,
        model_random_state=cfg.model_random_state,
        logger=_logger(cfg),
    )
    if tables is None:
        return

    if cfg.print_results:
        print_uncertainty_report(tables.across_seeds, cfg.uncertainty.alpha)

    out_dir = save_tables(
        dict(zip(UNCERTAINTY_TABLE_FILENAMES, tables)),
        cfg.uncertainty.output_dir,
        enabled=cfg.save_results,
    )
    if out_dir is not None and cfg.print_results:
        print(f"\nSaved uncertainty tables to: {out_dir.resolve()}")

    # `model` carries the interval method here: the fitted model is fixed by uncertainty_model, and what the run
    # compares is how each method's intervals behave.
    _, directory = save_results(
        wide_table_to_results(
            tables.by_split,
            {metric: metric for metric in UNCERTAINTY_METRICS},
            identifiers={
                "scenario": "scenario", "split_type": "split_type", "split_id": "split_id",
                "seed": "seed", "model": "method",
            },
        ),
        cfg.uncertainty.output_dir,
        enabled=cfg.save_results,
    )
    if directory is not None and cfg.print_results:
        print(f"Saved the unified result table to: {directory.resolve()}")
