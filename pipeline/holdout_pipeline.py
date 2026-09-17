"""Holdout orchestration: a train/valid split, scored once per seed, plus ablation."""

from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd

from config import RunConfig
from evaluate.metrics import METRICS, build_result_rows, compute_metrics
from interpret.case_studies import plot_case_studies
from interpret.model_weights import plot_permutation_importance, plot_shap_summary
from loaddata.splits import build_holdout_split
from loaddata.tabular_access import load_element_properties, load_feature_table
from pipeline.comparison import report_comparison
from pipeline.results import save_results
from utils.registry import ModelSpec, resolve_models
from utils.reporting import print_cv_results, print_holdout_results


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

    Which model to explain and which to compare come from the config (`interpret_model`, `case_study_models`).
    """
    pt, mm = load_element_properties(cfg.pt_path, cfg.mm_path)

    if cfg.ablation.interpret_model:
        (spec,) = resolve_models(registry, [cfg.ablation.interpret_model])
        if spec.key not in trained:
            raise ValueError(f"interpret_model {spec.key!r} is not in this run's models: {list(trained)}")
        model = trained[spec.key]
        plot_permutation_importance(
            model, X_valid, y_valid,
            title=f"{spec.name} Permutation Importance ({cfg.prefix})",
            save_path=plots_dir / f"{cfg.prefix}_perm_importance_{spec.key}.png",
        )
        plot_shap_summary(model, X_train, X_valid, save_path=plots_dir / f"{cfg.prefix}_shap_summary_{spec.key}.png")

    if cfg.ablation.case_study_models:
        specs = resolve_models(registry, cfg.ablation.case_study_models)
        missing = [spec.key for spec in specs if spec.key not in trained]
        if missing:
            raise ValueError(f"case_study_models names {missing}, which this run did not train: {list(trained)}")
        plot_case_studies(
            feature_columns,
            {spec.name: trained[spec.key] for spec in specs},
            pt,
            mm,
            target_label=cfg.target_column,
            save_path=plots_dir / f"{cfg.prefix}_case_studies.png",
        )


def run_holdout(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Repeat a train/validate split once per seed in holdout_seeds; report mean ± std.

    holdout_seeds seeds the split only, kept separate from random_state (model
    construction and tuning). Ablation plots use the first seed's models only.
    """
    specs = resolve_models(registry, cfg.models)
    plots_dir = Path(cfg.plots_output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    X, y, _ = load_feature_table(
        cfg.dataset_path,
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )
    feature_columns = list(X.columns)

    scores = {spec.name: {metric: [] for metric in METRICS} for spec in specs}
    collected = []
    first_split_models: Dict[str, object] = {}

    for run_i, seed in enumerate(cfg.holdout.seeds, start=1):
        print(f"\n=== Holdout Run {run_i}/{len(cfg.holdout.seeds)} (split seed={seed}) ===")

        _, train_idx, valid_idx = build_holdout_split(len(X), cfg.holdout.train_size, int(seed))
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        best_hyperparams = (
            _tune_on_split(specs, X_train, y_train, cfg)
            if cfg.tuning.enabled else {}
        )

        trained: Dict[str, object] = {}
        predictions = {}
        for spec in specs:
            model = spec.train(X_train, y_train, hyperparams=best_hyperparams.get(spec.key),
                               random_state=cfg.model_random_state)
            trained[spec.key] = model
            predictions[spec.name] = model.predict(X_valid)

        print_holdout_results(y_valid, predictions)

        per_seed = {name: compute_metrics(y_valid, y_pred) for name, y_pred in predictions.items()}
        for name, metrics in per_seed.items():
            for metric, value in metrics.items():
                scores[name][metric].append(value)
        collected.append(build_result_rows(
            {name: {m: [v] for m, v in metrics.items()} for name, metrics in per_seed.items()},
            scenario="holdout", split_type="ID", split_ids=[f"seed{seed}"], seed=int(seed),
        ))

        if not first_split_models:
            first_split_models = trained

    _, directory = save_results(
        pd.concat(collected, ignore_index=True),
        cfg.results_output_dir,
        prefix=f"{cfg.prefix}_holdout_",
        enabled=cfg.save_results,
    )
    if directory is not None:
        print(f"\nSaved holdout results to: {directory.resolve()}")

    if len(cfg.holdout.seeds) > 1:
        print(f"\n=== Holdout across {len(cfg.holdout.seeds)} splits ===")
        print_cv_results(scores, title="Holdout Metrics (mean ± std over split seeds):")
        if cfg.compare_models is not None:
            # One observation per seed, each a fresh split of the same rows, so the training sets overlap.
            report_comparison(
                cfg, registry, scores,
                test_train_ratio=(1.0 - cfg.holdout.train_size) / cfg.holdout.train_size,
            )

    if not cfg.ablation.enabled:
        return

    _, train_idx, valid_idx = build_holdout_split(len(X), cfg.holdout.train_size, int(cfg.holdout.seeds[0]))
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    _run_ablation(cfg, registry, first_split_models, feature_columns, X_train, X_valid, y_valid, plots_dir)
