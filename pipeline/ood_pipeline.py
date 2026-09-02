"""OOD evaluation orchestration: data loading, split selection, and reporting.

Reads the dataset CSV, asks pipeline/ood_scenarios.py which splits to build,
builds the size-matched in-distribution controls, hands all of that to
evaluate/ood_evaluation.py for scoring, and writes the result tables.

Deciding *which* splits exist lives here; scoring them lives in `evaluate/`.
That is why the controls are built in this module and passed down rather than
constructed inside the evaluator, which previously made `evaluate` import from
`pipeline` while `pipeline` imported from `evaluate`.
"""

from __future__ import annotations

import os
import zlib
from functools import partial
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from config import RunConfig
from core import ModelSpec, Split, resolve_models
from evaluate.ood_evaluation import (
    OOD,
    evaluate_splits_kfold_train_fixed_test,
    summarize_generalisation_gap,
    summarize_model_comparison,
    summarize_runs_across_splits,
)
from loaddata.feature_csv_access import resolve_feature_columns
from pipeline.ood_scenarios import build_scenarios
from pipeline.ood_splits import build_size_matched_split
from prepdata.alloy_transform import extract_elements_series, load_periodic_table_map
from reporting import print_ood_tables

# Result tables, in the order print_ood_tables takes them, paired with the file
# each is written to.
TABLE_FILENAMES = (
    "table1_splits_summary.csv",
    "table2_metrics_by_model.csv",
    "table3_model_comparison_significance.csv",
    "table4_combined_comparison.csv",
    "table5_generalisation_gap.csv",
)


def load_ood_dataset(
    train_dataset_path: str,
    test_dataset_path: Optional[str],
    target_column: str,
    formula_column: str,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the pool the OOD splits are carved out of.

    Both config paths are read and concatenated, because the OOD splits define
    their own train/test boundary — the two files are only a way of pointing at
    the data, not a pre-existing split that is honoured here.

    Args:
        train_dataset_path: First (or only) CSV to read.
        test_dataset_path: Second CSV, or the same path, or None.
        target_column: Name of the column being predicted.
        formula_column: Name of the chemical-formula column.
        feature_columns: Explicit feature list, or None to infer and validate
            them; see loaddata.feature_csv_access.resolve_feature_columns.

    Returns:
        (X, y, df_full), where X holds exactly the resolved feature columns.

    Raises:
        ValueError: If a required column is missing or the features do not
            resolve to a usable numeric matrix.
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

    resolved = resolve_feature_columns(df_full, target_column, formula_column, feature_columns)
    return df_full[resolved].copy(), df_full[target_column].copy(), df_full


def build_controls(
    splits: Sequence[Split],
    n_samples: int,
    seed: int,
    scenario: str,
) -> Dict[str, Split]:
    """One same-size random split per OOD split, keyed by the parent's split_id.

    crc32, not hash(): str hashing is salted per process, and these seeds have
    to be reproducible across runs.

    Returns:
        {split_id: control split}, omitting splits whose sizes do not fit.
    """
    controls: Dict[str, Split] = {}
    for split_id, train_idx, test_idx in splits:
        control = build_size_matched_split(
            n_samples, len(train_idx), len(test_idx),
            seed=zlib.crc32(f"{seed}|{scenario}|{split_id}".encode()),
            split_id=split_id,
        )
        if control is None:
            print(f"[WARN] No size-matched control fits for {scenario} {split_id}")
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
        print(f"[WARN] Skipping split {split_id} in {scenario}: {reason}")

    return _on_skip


def run_ood_evaluation(
    *,
    cfg: RunConfig,
    model_registry: Mapping[str, ModelSpec],
) -> None:
    """Run the OOD pipeline: build split families, score each, print and save tables.

    Called from main.py when evaluation_mode == 'ood'. Every OOD split is scored
    alongside its in-distribution references (see evaluate/ood_evaluation.py),
    so the saved tables carry a `split_type` column and Table 5 splits the
    degradation into a shift part and a training-pool part.

    Args:
        cfg: The resolved run configuration.
        model_registry: Registry to resolve `cfg.models` against.
    """
    ood_cfg = cfg.ood
    specs = resolve_models(model_registry, cfg.models)

    X_full, y_full, df_full = load_ood_dataset(
        cfg.train_dataset_path, cfg.test_dataset_path,
        cfg.target_column, cfg.formula_column, cfg.feature_columns,
    )
    element_to_group, element_to_period = load_periodic_table_map(cfg.pt_path)
    elements_per_row = extract_elements_series(df_full, formula_column=cfg.formula_column)

    scenarios = build_scenarios(
        ood_cfg, X_full, y_full, elements_per_row, element_to_group, element_to_period,
    )
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
            cv_folds=cfg.cv_folds, shuffle=cfg.cv_shuffle,
            hyperparameter_tuning=cfg.enable_hyperparameter_tuning, best_params=None,
            model_random_state=cfg.model_random_state,
            tune_cv_folds=cfg.tune_cv_folds, tune_n_iter=cfg.tune_n_iter,
            on_skip=_report_skip(scenario),
        )
        scenario_frames = _run_scenario(
            scenario, splits, seeds, len(X_full), ood_cfg.size_matched_control, evaluate,
        )
        for target, frames in zip(collected, scenario_frames):
            target.extend(frames)

    table1, table2 = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for frames in collected
    )

    table3 = pd.DataFrame()
    if cfg.compare_models is not None:
        name_a, name_b = (
            spec.name for spec in resolve_models(model_registry, cfg.compare_models)
        )
        table3 = summarize_model_comparison(table2, name_a, name_b, split_type=OOD)

    table4 = summarize_runs_across_splits(table2)
    table5 = summarize_generalisation_gap(table4)

    print_ood_tables(table1, table2, table3, table4, table5)

    out_dir = Path(ood_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for table, filename in zip((table1, table2, table3, table4, table5), TABLE_FILENAMES):
        if not table.empty:
            table.to_csv(out_dir / filename, index=False)

    print(f"\nSaved OOD tables to: {out_dir.resolve()}")
