"""Cross-validation orchestration: data loading, K-fold scoring, and reporting."""

from typing import Mapping

from config import RunConfig
from core import ModelSpec, resolve_models
from evaluate.cross_validation import cross_validate_models
from loaddata.feature_csv_access import load_features_and_target
from pipeline.comparison import report_comparison
from reporting import print_cv_results


def run_cross_validation(cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Run K-fold CV once per seed in cv_seeds and print the metrics."""
    specs = resolve_models(registry, cfg.models)
    X, y, _ = load_features_and_target(
        cfg.dataset_path,
        target_column=cfg.target_column,
        feature_columns=cfg.feature_columns,
        formula_column=cfg.formula_column,
    )

    if cfg.tuning.enabled:
        # Nested search multiplies fast; warn before spending an hour on it.
        tunable = [spec for spec in specs if spec.tune is not None]
        searches = len(cfg.kfold.seeds) * cfg.kfold.folds * len(tunable)
        print(
            f"\n[INFO] Nested hyperparameter search: {len(cfg.kfold.seeds)} seed(s) x "
            f"{cfg.kfold.folds} folds x {len(tunable)} tunable model(s) = {searches} searches, "
            f"each up to {cfg.tuning.n_iter} candidates x {cfg.tuning.cv_folds} inner folds "
            f"(~{searches * cfg.tuning.n_iter * cfg.tuning.cv_folds} model fits). "
            f"Lower tune_n_iter / tune_cv_folds in the config to shrink this."
        )

    for run_i, seed in enumerate(cfg.kfold.seeds, start=1):
        print(f"\n{'=' * 30}")
        print(f"=== CV Run {run_i}/{len(cfg.kfold.seeds)} (seed={seed}) ===")
        print(f"{'=' * 30}")

        results = cross_validate_models(
            X,
            y,
            specs,
            hyperparameter_tuning=cfg.tuning.enabled,
            best_params=None,
            cv_folds=cfg.kfold.folds,
            shuffle=cfg.kfold.shuffle,
            random_state=int(seed),
            model_random_state=cfg.model_random_state,
            tune_cv_folds=cfg.tuning.cv_folds,
            tune_n_iter=cfg.tuning.n_iter,
        )

        print_cv_results(results)
        if cfg.compare_models is not None:
            report_comparison(cfg, registry, results, cfg.kfold.folds)
