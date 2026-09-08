"""CLI entry point: run one evaluation mode against a YAML run config."""

import argparse
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from config import RunConfig, load_run_config
from core import ModelSpec, resolve_models
from loaddata.feature_csv_access import load_features_and_target, load_raw_data, split_dataset
from loaddata.raw_loaders import load_elemental_data

from pipeline.train import MODEL_REGISTRY
from pipeline.ood_pipeline import run_ood_evaluation
from pipeline.predict_pipeline import run_predict
from pipeline.uq_pipeline import run_uq_evaluation

from evaluate.cross_validation import (
    compare_models_significance,
    count_wins,
    cross_validate_models,
)
from evaluate.metrics import compute_metrics

from interpret.model_weights import plot_permutation_importance, plot_shap_summary
from interpret.case_studies import plot_case_studies
from interpret.visualize import plot_ms_distribution_by_tm, plot_violin_ms_by_tm, summarize_compound_radix

from reporting import (
    print_cv_results,
    print_holdout_results,
    print_significance,
    print_win_counts,
)

# Metrics the configured model pair is compared on.
COMPARISON_METRICS = ("mse", "mae")


def parse_args() -> argparse.Namespace:
    """Parse the --config CLI flag."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML models for material property prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/novamag.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def _comparison_names(
    cfg: RunConfig, registry: Mapping[str, ModelSpec]
) -> Tuple[str, str]:
    """Resolve `compare_models` to the two display names used in result tables."""
    specs = resolve_models(registry, cfg.compare_models)
    return specs[0].name, specs[1].name


def _report_comparison(cfg: RunConfig, registry, results, n_repeats: int) -> None:
    """Print the win tally and paired tests for the configured model pair."""
    name_a, name_b = _comparison_names(cfg, registry)
    print_win_counts(count_wins(results, name_a, name_b, metric="mse"), n_repeats)
    for metric in COMPARISON_METRICS:
        print_significance(compare_models_significance(results, name_a, name_b, metric=metric))


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
            _report_comparison(cfg, registry, results, cfg.kfold.folds)


def _tune_on_split(specs, X_train, y_train, cfg: RunConfig) -> Dict[str, Dict]:
    """Search hyperparameters on one split's training set."""
    print("--- Hyperparameter Tuning (on this split's training set) ---")
    return {
        spec.key: spec.tune(
            X_train,
            y_train,
            cv_folds=cfg.tuning.cv_folds,
            random_state=cfg.model_random_state,
            n_iter=cfg.tuning.n_iter,
        )
        for spec in specs
        if spec.tune is not None
    }


def _run_ablation(
    cfg: RunConfig,
    registry: Mapping[str, ModelSpec],
    trained: Mapping[str, object],
    feature_columns: Sequence[str],
    X_train,
    X_valid,
    y_valid,
    plots_dir: Path,
) -> None:
    """Produce the interpretability figures for one set of fitted models.

    Which model to explain and which to compare come from the config
    (`interpret_model`, `case_study_models`).
    """
    pt, mm = load_elemental_data(cfg.pt_path, cfg.mm_path)

    if cfg.interpret_model:
        (spec,) = resolve_models(registry, [cfg.interpret_model])
        if spec.key not in trained:
            raise ValueError(
                f"interpret_model {spec.key!r} is not in this run's models: {list(trained)}"
            )
        model = trained[spec.key]
        plot_permutation_importance(
            model, X_valid, y_valid,
            title=f"{spec.name} Permutation Importance ({cfg.prefix})",
            save_path=plots_dir / f"{cfg.prefix}_perm_importance_{spec.key}.png",
        )
        plot_shap_summary(
            model, X_train, X_valid,
            save_path=plots_dir / f"{cfg.prefix}_shap_summary_{spec.key}.png",
        )

    if cfg.case_study_models:
        specs = resolve_models(registry, cfg.case_study_models)
        missing = [spec.key for spec in specs if spec.key not in trained]
        if missing:
            raise ValueError(
                f"case_study_models names {missing}, which this run did not train: {list(trained)}"
            )
        plot_case_studies(
            feature_columns,
            {spec.name: trained[spec.key] for spec in specs},
            pt,
            mm,
            target_label=cfg.target_column,
            save_path=plots_dir / f"{cfg.prefix}_case_studies.png",
        )


def run_holdout(cfg: RunConfig, registry: Mapping[str, ModelSpec], plots_dir: Path) -> None:
    """Repeat a train/validate split once per seed in holdout_seeds; report mean ± std.

    holdout_seeds seeds the split only, kept separate from random_state (model
    construction and tuning). Ablation plots use the first seed's models only.
    """
    specs = resolve_models(registry, cfg.models)
    X, y, feature_columns = load_features_and_target(
        cfg.dataset_path,
        target_column=cfg.target_column,
        feature_columns=cfg.feature_columns,
        formula_column=cfg.formula_column,
    )

    scores = {spec.name: {"mse": [], "mae": [], "mre": [], "r2": []} for spec in specs}
    first_split_models: Dict[str, object] = {}

    for run_i, seed in enumerate(cfg.holdout.seeds, start=1):
        print(f"\n=== Holdout Run {run_i}/{len(cfg.holdout.seeds)} (split seed={seed}) ===")

        X_train, X_valid, y_train, y_valid = split_dataset(
            X, y, train_size=cfg.holdout.train_size, random_state=int(seed)
        )

        best_params = (
            _tune_on_split(specs, X_train, y_train, cfg)
            if cfg.tuning.enabled else {}
        )

        trained: Dict[str, object] = {}
        predictions = {}
        for spec in specs:
            model = spec.train(
                X_train, y_train,
                params=best_params.get(spec.key),
                random_state=cfg.model_random_state,
            )
            trained[spec.key] = model
            predictions[spec.name] = model.predict(X_valid)

        print_holdout_results(y_valid, predictions)

        for name, y_pred in predictions.items():
            for metric, value in compute_metrics(y_valid, y_pred).items():
                scores[name][metric].append(value)

        if not first_split_models:
            first_split_models = trained

    if len(cfg.holdout.seeds) > 1:
        print(f"\n=== Holdout across {len(cfg.holdout.seeds)} splits ===")
        print_cv_results(scores, title="Holdout Metrics (mean ± std over split seeds):")
        if cfg.compare_models is not None:
            _report_comparison(cfg, registry, scores, len(cfg.holdout.seeds))

    if not cfg.enable_ablation_study:
        return

    X_train, X_valid, y_train, y_valid = split_dataset(
        X, y, train_size=cfg.holdout.train_size, random_state=int(cfg.holdout.seeds[0])
    )
    _run_ablation(
        cfg, registry, first_split_models, feature_columns,
        X_train, X_valid, y_valid, plots_dir,
    )


def run_data_visualization(cfg: RunConfig, plots_dir: Path) -> None:
    """Plot the target's distribution across the transition-metal subsets."""
    raw = load_raw_data(cfg.dataset_path)
    plot_ms_distribution_by_tm(
        raw, save_path=plots_dir / f"{cfg.prefix}_ms_distribution_by_tm.png"
    )
    plot_violin_ms_by_tm(
        raw,
        title=f"{cfg.prefix.upper()} Violin Plot",
        save_path=plots_dir / f"{cfg.prefix}_violin_ms_by_tm.png",
    )
    summarize_compound_radix(raw)


def main() -> None:
    """Run one predict/holdout/cross_validation/ood/uq job, per the --config file."""
    args = parse_args()
    cfg = load_run_config(args.config)

    plots_dir = Path(cfg.plots_output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if cfg.evaluation_mode == "predict":
        run_predict(cfg=cfg, model_registry=MODEL_REGISTRY)
        return

    if cfg.evaluation_mode == "cross_validation":
        run_cross_validation(cfg, MODEL_REGISTRY)
    elif cfg.evaluation_mode == "ood":
        run_ood_evaluation(cfg=cfg, model_registry=MODEL_REGISTRY)
    elif cfg.evaluation_mode == "uq":
        run_uq_evaluation(cfg=cfg, model_registry=MODEL_REGISTRY)
    else:
        run_holdout(cfg, MODEL_REGISTRY, plots_dir)

    # Data visualization reads the raw dataset, which OOD mode does not require.
    if cfg.enable_data_visualization and cfg.evaluation_mode != "ood":
        run_data_visualization(cfg, plots_dir)


if __name__ == "__main__":
    main()
