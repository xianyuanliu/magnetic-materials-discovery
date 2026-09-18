"""Config adapter for `evaluation_mode: cross_validation`."""

from typing import Mapping

from config import RunConfig
from loaddata.tabular_access import load_feature_table
from pipeline.cross_validation import run_cross_validation
from utils.persistence import save_results
from utils.registry import ModelSpec, resolve_models
from utils.reporting import SILENT, Logger, print_summary


def run(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Load what the config names, run repeated K-fold, then save and print what it asks for.

    Args:
        cfg: The resolved run configuration.
        registry: Registry to resolve `models` and `compare_models` against.
    """
    specs = resolve_models(registry, cfg.models)
    X, y, _ = load_feature_table(
        cfg.dataset_path,
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )

    # Resolved once: the pair is the same for every seed.
    compared = (
        tuple(spec.name for spec in resolve_models(registry, cfg.compare_models))
        if cfg.compare_models is not None else None
    )

    results = run_cross_validation(
        X, y, specs,
        folds=cfg.kfold.folds,
        shuffle=cfg.kfold.shuffle,
        seeds=cfg.kfold.seeds,
        hyperparameter_tuning=cfg.tuning.enabled,
        tune_cv_folds=cfg.tuning.cv_folds,
        tune_n_iter=cfg.tuning.n_iter,
        model_random_state=cfg.model_random_state,
        compare=compared,
        logger=Logger() if cfg.print_results else SILENT,
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
