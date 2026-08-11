"""OOD evaluation orchestration: data loading, split selection, and reporting.

Reads the dataset CSV, asks pipeline/ood_scenarios.py which splits to build,
scores each via evaluate/ood_evaluation.py, and writes the result tables.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from evaluate.ood_evaluation import (
    evaluate_splits_kfold_train_fixed_test,
    print_ood_tables,
    summarize_generalisation_gap,
    summarize_runs_across_splits,
)
from pipeline.ood_scenarios import build_scenarios, load_ood_config
from pipeline.ood_splits import Split
from pipeline.train import DEFAULT_TUNE_CV_FOLDS, DEFAULT_TUNE_N_ITER
from prepdata.alloy_transform import extract_elements_series, load_periodic_table_map

# Result tables, in the order print_ood_tables takes them, paired with the file
# each is written to.
TABLE_FILENAMES = (
    "table1_splits_summary.csv",
    "table2_metrics_by_model.csv",
    "table3_rf_vs_xgb_significance.csv",
    "table4_combined_comparison.csv",
    "table5_generalisation_gap.csv",
)


def load_ood_dataset(
    train_dataset_path: str,
    test_dataset_path: Optional[str],
    target_column: str,
    formula_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the pool the OOD splits are carved out of.

    Both config paths are read and concatenated, because the OOD splits define
    their own train/test boundary — the two files are only a way of pointing at
    the data, not a pre-existing split that is honoured here.

    Returns:
        (X, y, df_full), where X excludes the target and formula columns.

    Raises:
        ValueError: If a required column is missing or no feature column is left.
    """
    if not train_dataset_path:
        raise ValueError("OOD mode requires train_dataset_path (can be the full dataset CSV).")

    same_file = (not test_dataset_path) or (
        os.path.normpath(test_dataset_path) == os.path.normpath(train_dataset_path)
    )
    if same_file:
        df_full = pd.read_csv(train_dataset_path).reset_index(drop=True)
    else:
        df_full = pd.concat(
            [pd.read_csv(train_dataset_path), pd.read_csv(test_dataset_path)],
            ignore_index=True,
        )

    for column in (target_column, formula_column):
        if column not in df_full.columns:
            raise ValueError(f"Missing column '{column}' in the dataset CSV(s)")

    feature_cols = [c for c in df_full.columns if c not in (target_column, formula_column)]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding target/formula columns.")

    return df_full[feature_cols].copy(), df_full[target_column].copy(), df_full


def _name_for_key(model_registry: Dict, model_key: str) -> str:
    return str(model_registry[model_key]["name"])


def _run_scenario(
    scenario: str,
    splits: List[Split],
    seeds: Sequence[int],
    evaluate,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """Evaluate one scenario's splits across all seeds.

    Returns:
        (t1_frames, t2_frames, t3_frames) — one triple of tables per seed.
    """
    frames: Tuple[List, List, List] = ([], [], [])
    for seed in seeds:
        for collected, table in zip(frames, evaluate(splits, scenario=scenario, seed=int(seed))):
            collected.append(table)
    return frames


def run_ood_evaluation(
    *,
    cfg: dict,
    train_dataset_path: str,
    test_dataset_path: str,
    pt_path: str,
    models: List[str],
    model_registry: Dict,
    cv_folds: int,
    cv_shuffle: bool,
    hyperparameter_tuning: bool,
    cv_random_state: int,
    model_random_state: int = 0,
    tune_cv_folds: int = DEFAULT_TUNE_CV_FOLDS,
    tune_n_iter: int = DEFAULT_TUNE_N_ITER,
) -> None:
    """Run the OOD pipeline: build split families, score each, print and save tables.

    Called from main.py when evaluation_mode == 'ood'. Every OOD split is scored
    alongside its in-distribution references (see evaluate/ood_evaluation.py),
    so the saved tables carry a `split_type` column and Table 5 splits the
    degradation into a shift part and a training-pool part.
    """
    ood_cfg = load_ood_config(cfg, default_seed=int(cv_random_state))

    X_full, y_full, df_full = load_ood_dataset(
        train_dataset_path, test_dataset_path, ood_cfg.target_column, ood_cfg.formula_column,
    )
    element_to_group, element_to_period = load_periodic_table_map(pt_path)
    elements_per_row = extract_elements_series(df_full, formula_column=ood_cfg.formula_column)

    scenarios = build_scenarios(
        ood_cfg, X_full, y_full, elements_per_row, element_to_group, element_to_period,
    )
    seeds = list(ood_cfg.cv_seeds) if ood_cfg.cv_seeds is not None else [int(ood_cfg.ood_seed)]

    print(
        f"\n[INFO] OOD run: {len(scenarios)} scenario(s), "
        f"{sum(len(s) for _, s in scenarios)} split(s), {len(seeds)} seed(s), "
        f"size-matched control {'on' if ood_cfg.size_matched_control else 'off'}."
    )
    for name, splits in scenarios:
        print(f"         {name}: {len(splits)} split(s) — {[s[0] for s in splits]}")

    evaluate = partial(
        evaluate_splits_kfold_train_fixed_test,
        X_full, y_full,
        model_keys=models, model_registry=model_registry,
        cv_folds=cv_folds, shuffle=cv_shuffle,
        hyperparameter_tuning=hyperparameter_tuning, best_params=None,
        model_random_state=model_random_state,
        rf_name=_name_for_key(model_registry, "rf") if "rf" in models else None,
        xgb_name=_name_for_key(model_registry, "xgb") if "xgb" in models else None,
        tune_cv_folds=tune_cv_folds, tune_n_iter=tune_n_iter,
        size_matched_control=ood_cfg.size_matched_control,
    )

    collected: Tuple[List, List, List] = ([], [], [])
    for scenario, splits in scenarios:
        for target, frames in zip(collected, _run_scenario(scenario, splits, seeds, evaluate)):
            target.extend(frames)

    table1, table2, table3 = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for frames in collected
    )
    table4 = summarize_runs_across_splits(table2)
    table5 = summarize_generalisation_gap(table4)

    print_ood_tables(table1, table2, table3, table4, table5)

    out_dir = Path(ood_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for table, filename in zip((table1, table2, table3, table4, table5), TABLE_FILENAMES):
        if not table.empty:
            table.to_csv(out_dir / filename, index=False)

    print(f"\nSaved OOD tables to: {out_dir.resolve()}")
