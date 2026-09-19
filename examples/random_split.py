"""Config adapter for `evaluation_mode: random_split`, including the interpretability figures it can ask for."""

from pathlib import Path
from typing import Mapping, Sequence

from config import RunConfig
from evaluate.metrics import results_to_scores
from interpret.case_studies import plot_case_studies
from interpret.model_explanations import plot_model_explanations
from loaddata.splits import build_random_split
from loaddata.tabular_access import load_element_properties, load_feature_table
from pipeline.random_split import run_random_split
from utils.persistence import save_results
from utils.registry import ModelSpec, resolve_models, resolve_trained_models
from utils.reporting import SILENT, Logger, format_comparisons, print_summary


def _plot_ablation(
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

    if cfg.ablation.interpret_model:
        (spec,) = resolve_trained_models(registry, trained, [cfg.ablation.interpret_model])
        plot_model_explanations(
            trained[spec.key], X_train, X_valid, y_valid,
            name=spec.name, save_dir=plots_dir, prefix=f"{cfg.prefix}_",
            random_state=cfg.model_random_state,
        )

    if cfg.ablation.case_study_models:
        specs = resolve_trained_models(registry, trained, cfg.ablation.case_study_models)
        pt, mm = load_element_properties(cfg.pt_path, cfg.mm_path)
        plot_case_studies(
            feature_columns,
            {spec.name: trained[spec.key] for spec in specs},
            pt,
            mm,
            target_label=cfg.target_column,
            save_path=plots_dir / f"{cfg.prefix}_case_studies.png",
        )


def run(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Load what the config names, repeat the split once per seed, then save, print and explain what it asks for.

    `random_split_seeds` seeds the split only, kept separate from `random_state` (model construction and tuning).
    Ablation figures come from the first seed's models, so they stay put when a seed is added.

    Args:
        cfg: The resolved run configuration.
        registry: Registry to resolve `models`, `compare_models` and the ablation model names against.
    """
    specs = resolve_models(registry, cfg.models)
    X, y, _ = load_feature_table(
        cfg.dataset_path,
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )

    results, first_split_models = run_random_split(
        X, y, specs,
        seeds=cfg.random_split.seeds,
        train_size=cfg.random_split.train_size,
        hyperparameter_tuning=cfg.tuning.enabled,
        tune_cv_folds=cfg.tuning.cv_folds,
        tune_n_iter=cfg.tuning.n_iter,
        model_random_state=cfg.model_random_state,
        logger=Logger() if cfg.print_results else SILENT,
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
        if cfg.compare_models is not None:
            # One observation per seed, each a fresh split of the same rows, so the training sets overlap.
            name_a, name_b = (spec.name for spec in resolve_models(registry, cfg.compare_models))
            print(format_comparisons(
                results_to_scores(results), name_a, name_b,
                test_train_ratio=(1.0 - cfg.random_split.train_size) / cfg.random_split.train_size,
            ), end="")

    if not cfg.ablation.enabled:
        return

    # The same split the models came from, rebuilt rather than carried: build_random_split is pure and seeded.
    _, train_idx, valid_idx = build_random_split(len(X), cfg.random_split.train_size, int(cfg.random_split.seeds[0]))
    _plot_ablation(
        cfg, registry, first_split_models, list(X.columns),
        X.iloc[train_idx], X.iloc[valid_idx], y.iloc[valid_idx],
    )
