"""Cross-validation orchestration: data loading, K-fold scoring, and reporting."""

from typing import Mapping

import pandas as pd

from config import RunConfig
from evaluate.cross_validation import cross_validate_models, scores_to_results
from loaddata.splits import build_kfold_splits
from loaddata.tabular_access import load_feature_table
from utils.persistence import save_results
from utils.registry import ModelSpec, resolve_models
from utils.reporting import print_comparisons, print_cv_results, print_search_budget, print_summary


def run_cross_validation(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Run K-fold CV once per seed in cv_seeds and print the metrics."""
    specs = resolve_models(registry, cfg.models)
    X, y, _ = load_feature_table(
        cfg.dataset_path,
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )

    if cfg.tuning.enabled and cfg.print_results:
        print_search_budget(
            len(cfg.kfold.seeds), cfg.kfold.folds,
            sum(1 for spec in specs if spec.tune is not None),
            cfg.tuning.n_iter, cfg.tuning.cv_folds,
        )

    # Resolved once: the pair is the same for every seed.
    compared = (
        tuple(spec.name for spec in resolve_models(registry, cfg.compare_models))
        if cfg.compare_models is not None else None
    )

    collected = []
    for run_i, seed in enumerate(cfg.kfold.seeds, start=1):
        if cfg.print_results:
            print(f"\n=== CV Run {run_i}/{len(cfg.kfold.seeds)} (seed={seed}) ===")

        splits = build_kfold_splits(len(X), cfg.kfold.folds, cfg.kfold.shuffle, int(seed))
        results = cross_validate_models(
            X,
            y,
            specs,
            splits,
            hyperparameter_tuning=cfg.tuning.enabled,
            best_hyperparams=None,
            model_random_state=cfg.model_random_state,
            tune_cv_folds=cfg.tuning.cv_folds,
            tune_n_iter=cfg.tuning.n_iter,
        )
        collected.append(scores_to_results(results, splits, scenario="cross_validation", seed=int(seed)))

        if cfg.print_results:
            print_cv_results(results)
        if compared is not None and cfg.print_results:
            # Every fold trains on all but one of K parts, so one fold's test set is 1 / (K - 1) of its
            # training set. The folds share training data; compare_models_significance corrects for it.
            print_comparisons(results, *compared, test_train_ratio=1.0 / (cfg.kfold.folds - 1))

    summary, directory = save_results(
        pd.concat(collected, ignore_index=True),
        cfg.results_output_dir,
        prefix=f"{cfg.prefix}_cv_",
        enabled=cfg.save_results,
    )
    # Across every fold of every seed, so cross-validation and holdout close on the same statement.
    if cfg.print_results and len(cfg.kfold.seeds) > 1:
        print_summary(summary, f"Cross-Validation Metrics across {len(cfg.kfold.seeds)} seed(s) (mean ± std):")
    if directory is not None and cfg.print_results:
        print(f"\nSaved cross-validation results to: {directory.resolve()}")
