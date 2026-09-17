"""OOD evaluation orchestration: data loading, split selection, and reporting.

Reads the dataset CSV, asks pipeline/ood_scenarios.py which splits to build, builds the size-matched in-distribution
controls, hands all of that to evaluate/ood_evaluation.py for scoring, and writes the result tables.

Deciding *which* splits exist lives here; scoring them lives in `evaluate/`. The size-matched controls and the inner
K-fold over each split's training rows are therefore both built here and passed down, so the evaluator never has to
know what a scenario is.
"""

from __future__ import annotations

import warnings
import zlib
from functools import partial
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from config import RunConfig
from evaluate.metrics import METRICS, melt_to_results
from evaluate.ood_evaluation import (
    OOD,
    evaluate_splits_kfold_train_fixed_test,
    summarize_generalization_gap,
    summarize_model_comparison,
    summarize_runs_across_splits,
)
from loaddata.splits import Split, build_kfold_splits, build_size_matched_split
from loaddata.tabular_access import load_feature_table
from pipeline.ood_scenarios import build_scenarios, resolve_split_elements
from utils.persistence import save_results, save_tables
from utils.registry import ModelSpec, resolve_models
from utils.reporting import print_ood_tables

# Result tables, in the order print_ood_tables takes them, paired with the file each is written to.
TABLE_FILENAMES = (
    "table1_splits_summary.csv",
    "table2_metrics_by_model.csv",
    "table3_model_comparison_significance.csv",
    "table4_combined_comparison.csv",
    "table5_generalization_gap.csv",
)


def build_controls(splits: Sequence[Split], n_samples: int, seed: int, scenario: str) -> Dict[str, Split]:
    """One same-size random split per OOD split, keyed by the parent's split_id.

    Returns:
        {split_id: control split}, omitting splits whose sizes do not fit.
    """
    controls: Dict[str, Split] = {}
    for split_id, train_idx, test_idx in splits:
        control = build_size_matched_split(
            n_samples, len(train_idx), len(test_idx),
            # crc32, not hash(): str hashing is salted per process, and these seeds must be reproducible across runs.
            seed=zlib.crc32(f"{seed}|{scenario}|{split_id}".encode()),
            split_id=split_id,
        )
        if control is None:
            warnings.warn(f"No size-matched control fits for {scenario} {split_id}.")
        else:
            controls[split_id] = control
    return controls


def _run_scenario(
    scenario: str,
    splits: List[Split],
    seeds: Sequence[int],
    n_samples: int,
    size_matched_control: bool,
    evaluate,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Evaluate one scenario's splits across all seeds.

    Returns:
        (t1_frames, t2_frames) — one pair of tables per seed.
    """
    frames: Tuple[List, List] = ([], [])
    for seed in seeds:
        controls = (
            build_controls(splits, n_samples, int(seed), scenario)
            if size_matched_control else None
        )
        tables = evaluate(splits, scenario=scenario, seed=int(seed), controls=controls)
        for collected, table in zip(frames, tables):
            collected.append(table)
    return frames


def _report_skip(scenario: str):
    """Return an on_skip callback that labels the scenario it came from."""
    def _on_skip(split_id: str, reason: str) -> None:
        warnings.warn(f"Skipping split {split_id} in {scenario}: {reason}")

    return _on_skip


def run_ood_evaluation(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Run the OOD pipeline: build split families, score each, print and save tables.

    Called from main.py when evaluation_mode == 'ood'. Every OOD split is scored alongside its in-distribution
    references (see evaluate/ood_evaluation.py), so the saved tables carry a `split_type` column and Table 5 splits the
    degradation into a shift part and a training-pool part.

    Args:
        cfg: The resolved run configuration.
        registry: Registry to resolve `cfg.models` against.
    """
    ood_cfg = cfg.ood
    specs = resolve_models(registry, cfg.models)

    X_full, y_full, metadata = load_feature_table(
        [cfg.train_dataset_path, cfg.test_dataset_path],
        target_column=cfg.target_column,
        formula_column=cfg.formula_column,
        feature_columns=cfg.feature_columns,
    )
    elements_per_row, element_to_group, element_to_period = resolve_split_elements(
        metadata, cfg.formula_column, cfg.pt_path
    )

    scenarios = build_scenarios(ood_cfg, X_full, y_full, elements_per_row, element_to_group, element_to_period)
    seeds = list(ood_cfg.seeds)

    print(
        f"\n[INFO] OOD run: {len(scenarios)} scenario(s), "
        f"{sum(len(s) for _, s in scenarios)} split(s), {len(seeds)} seed(s), "
        f"size-matched control {'on' if ood_cfg.size_matched_control else 'off'}."
    )
    for name, splits in scenarios:
        print(f"         {name}: {len(splits)} split(s) — {[s[0] for s in splits]}")

    collected: Tuple[List, List] = ([], [])
    for scenario, splits in scenarios:
        evaluate = partial(
            evaluate_splits_kfold_train_fixed_test,
            X_full, y_full, specs=specs,
            inner_splitter=lambda n_train, seed: build_kfold_splits(
                n_train, cfg.kfold.folds, cfg.kfold.shuffle, seed
            ),
            hyperparameter_tuning=cfg.tuning.enabled, best_hyperparams=None,
            model_random_state=cfg.model_random_state,
            tune_cv_folds=cfg.tuning.cv_folds, tune_n_iter=cfg.tuning.n_iter,
            on_skip=_report_skip(scenario),
        )
        scenario_frames = _run_scenario(scenario, splits, seeds, len(X_full), ood_cfg.size_matched_control, evaluate)
        for target, frames in zip(collected, scenario_frames):
            target.extend(frames)

    splits_summary, metrics_by_model = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for frames in collected
    )

    comparison_significance = pd.DataFrame()
    if cfg.compare_models is not None:
        name_a, name_b = (
            spec.name for spec in resolve_models(registry, cfg.compare_models)
        )
        comparison_significance = summarize_model_comparison(metrics_by_model, name_a, name_b, split_type=OOD)

    combined_comparison = summarize_runs_across_splits(metrics_by_model)
    generalization_gap = summarize_generalization_gap(combined_comparison)

    tables = (
        splits_summary, metrics_by_model, comparison_significance,
        combined_comparison, generalization_gap,
    )
    if cfg.print_results:
        print_ood_tables(*tables)

    out_dir = save_tables(dict(zip(TABLE_FILENAMES, tables)), ood_cfg.output_dir, enabled=cfg.save_results)
    if out_dir is not None and cfg.print_results:
        print(f"\nSaved OOD tables to: {out_dir.resolve()}")

    # The unified table reports one observation per split, matching table 2's *_mean columns: the inner folds all score
    # the same held-out rows, so their spread belongs in *_std rather than as separate observations.
    _, directory = save_results(
        melt_to_results(
            metrics_by_model,
            {f"{metric.upper()}_mean": metric for metric in METRICS},
            identifiers={
                "scenario": "scenario", "split_type": "split_type", "split_id": "split_id",
                "seed": "seed", "model": "model",
            },
        ),
        ood_cfg.output_dir,
        enabled=cfg.save_results,
    )
    if directory is not None and cfg.print_results:
        print(f"Saved the unified result table to: {directory.resolve()}")
