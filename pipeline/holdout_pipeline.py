"""Holdout orchestration: a train/valid split, scored once per seed, plus ablation."""

from pathlib import Path
from typing import Dict, Mapping, Sequence

from config import RunConfig
from core import ModelSpec, resolve_models
from evaluate.metrics import compute_metrics
from interpret.case_studies import plot_case_studies
from interpret.model_weights import plot_permutation_importance, plot_shap_summary
from loaddata.feature_csv_access import load_features_and_target, split_dataset
from loaddata.raw_loaders import load_elemental_data
from pipeline.comparison import report_comparison
from reporting import print_cv_results, print_holdout_results


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
            report_comparison(cfg, registry, scores)

    if not cfg.enable_ablation_study:
        return

    X_train, X_valid, y_train, y_valid = split_dataset(
        X, y, train_size=cfg.holdout.train_size, random_state=int(cfg.holdout.seeds[0])
    )
    _run_ablation(
        cfg, registry, first_split_models, feature_columns,
        X_train, X_valid, y_valid, plots_dir,
    )
