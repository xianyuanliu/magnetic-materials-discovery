"""Random-split orchestration: a train/valid split drawn at random, scored once per seed, plus ablation."""

from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd

from config import RunConfig
from evaluate.metrics import METRICS, build_result_rows, compute_metrics
from interpret.case_studies import plot_case_studies
from interpret.model_explanations import plot_model_explanations
from loaddata.splits import build_random_split
from loaddata.tabular_access import load_element_properties, load_feature_table
from utils.persistence import save_results
from utils.registry import ModelSpec, resolve_models
from utils.reporting import print_comparisons, print_split_results, print_summary


def _tune_on_split(specs, X_train, y_train, cfg: RunConfig) -> Dict[str, Dict]:
    """Search hyperparameters on one split's training set."""
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
    """Resolve which fitted models the config asks to explain, and hand them to `interpret/`.

    Only the resolution is here: turning `interpret_model` and `case_study_models` into fitted estimators, and failing
    loudly when they name a model this run never trained. The explanations themselves are config-free.
    """
    def _resolve(keys: Sequence[str]) -> Sequence[ModelSpec]:
        specs = resolve_models(registry, keys)
        missing = [spec.key for spec in specs if spec.key not in trained]
        if missing:
            raise ValueError(f"{missing} named for interpretation, but this run trained {list(trained)}")
        return specs

    if cfg.ablation.interpret_model:
        (spec,) = _resolve([cfg.ablation.interpret_model])
        plot_model_explanations(
            trained[spec.key], X_train, X_valid, y_valid,
            name=spec.name, save_dir=plots_dir, prefix=f"{cfg.prefix}_",
            random_state=cfg.model_random_state,
        )

    if cfg.ablation.case_study_models:
        pt, mm = load_element_properties(cfg.pt_path, cfg.mm_path)
        plot_case_studies(
            feature_columns,
            {spec.name: trained[spec.key] for spec in _resolve(cfg.ablation.case_study_models)},
            pt,
            mm,
            target_label=cfg.target_column,
            save_path=plots_dir / f"{cfg.prefix}_case_studies.png",
        )


def run_random_split(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Repeat a train/validate split once per seed in random_split_seeds; report mean ± std.

    random_split_seeds seeds the split only, kept separate from random_state (model
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

    for run_i, seed in enumerate(cfg.random_split.seeds, start=1):
        if cfg.print_results:
            print(f"\n=== Random Split Run {run_i}/{len(cfg.random_split.seeds)} (split seed={seed}) ===")

        _, train_idx, valid_idx = build_random_split(len(X), cfg.random_split.train_size, int(seed))
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

        if cfg.print_results:
            print_split_results(y_valid, predictions)

        per_seed = {name: compute_metrics(y_valid, y_pred) for name, y_pred in predictions.items()}
        for name, metrics in per_seed.items():
            for metric, value in metrics.items():
                scores[name][metric].append(value)
        collected.append(build_result_rows(
            {name: {m: [v] for m, v in metrics.items()} for name, metrics in per_seed.items()},
            scenario="random_split", split_type="ID", split_ids=[f"seed{seed}"], seed=int(seed),
        ))

        if not first_split_models:
            first_split_models = trained

    summary, directory = save_results(
        pd.concat(collected, ignore_index=True),
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
            print_comparisons(
                scores, name_a, name_b,
                test_train_ratio=(1.0 - cfg.random_split.train_size) / cfg.random_split.train_size,
            )

    if not cfg.ablation.enabled:
        return

    _, train_idx, valid_idx = build_random_split(len(X), cfg.random_split.train_size, int(cfg.random_split.seeds[0]))
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    _run_ablation(cfg, registry, first_split_models, feature_columns, X_train, X_valid, y_valid, plots_dir)
