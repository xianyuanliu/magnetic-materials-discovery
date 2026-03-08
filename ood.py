"""
OOD evaluation orchestration.

Design goals:

- uses a configurable random seed for OOD split generation

- reads train/test CSVs (fixed split) and runs extra OOD stress tests on FULL data
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd

import os

from evaluate import (
    evaluate_splits_kfold_train_fixed_test,
    summarize_runs_across_splits,
    print_ood_tables,
)

# Uses your existing split builders (recommended to keep as a separate, auditable file)
from ood_splits import (
    build_loeo_splits,
    build_period_splits,
    build_group_splits,
    build_kmeans_cluster_splits,
    build_sparsex_splits,
    build_sparsey_splits,
)

# These should already exist from your PR5 utilities commit
from data import (
    extract_elements_series,
    load_periodic_table_map,
)

Split = Tuple[str, np.ndarray, np.ndarray]


@dataclass
class OODConfig:
    target_column: str = "saturation magnetization"
    formula_column: str = "chemical formula"

    # which OOD scenarios to run: element | period | group | cluster | all
    ood_mode: str = "all"

    # LOCO settings
    ood_k: int = 10

    # selection / limits
    ood_seed: int = 0
    ood_max_splits: int = 10
    ood_targets: Optional[Sequence[Any]] = None  # list of elements OR periods OR groups (depending on mode)
    ood_fractions: Sequence[float] = (0.1, 0.2)
    sparsex_neighbors: int = 5
    sparsey_center: str = "median"
    cv_seeds: Optional[Sequence[int]] = None

    # strict settings for period/group (optional)
    period_strict: bool = False
    group_strict: bool = False

    # output (optional)
    output_dir: str = "./results/ood"


def _safe_int_list(x: Optional[Sequence[Any]]) -> Optional[List[int]]:
    if x is None:
        return None
    return [int(v) for v in x]


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _top_elements(elements_per_row: Sequence[Sequence[str]], max_n: int) -> List[str]:
    counts: Dict[str, int] = {}
    for els in elements_per_row:
        for e in set(els):
            counts[e] = counts.get(e, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [e for e, _ in ordered[:max_n]]


def _top_periods(
    elements_per_row: Sequence[Sequence[str]],
    element_to_period: Dict[str, int],
    max_n: int,
) -> List[int]:
    counts: Dict[int, int] = {}
    for els in elements_per_row:
        periods = set()
        for e in set(els):
            p = element_to_period.get(e)
            if p is not None:
                periods.add(int(p))
        for p in periods:
            counts[p] = counts.get(p, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [int(p) for p, _ in ordered[:max_n]]


def _top_groups(
    elements_per_row: Sequence[Sequence[str]],
    element_to_group: Dict[str, int],
    max_n: int,
) -> List[int]:
    counts: Dict[int, int] = {}
    for els in elements_per_row:
        groups = set()
        for e in set(els):
            g = element_to_group.get(e)
            if g is not None:
                groups.add(int(g))
        for g in groups:
            counts[g] = counts.get(g, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [int(g) for g, _ in ordered[:max_n]]


def _name_for_key(model_registry: Dict, model_key: str) -> str:
    return str(model_registry[model_key]["name"])


def _load_ood_cfg(cfg: dict, default_seed: int) -> OODConfig:
    return OODConfig(
        target_column=str(cfg.get("target_column", "saturation magnetization")),
        formula_column=str(cfg.get("formula_column", "chemical formula")),
        ood_mode=str(cfg.get("ood_mode", "all")).lower(),
        ood_k=int(cfg.get("ood_k", 10)),
        ood_seed=int(cfg.get("ood_seed", default_seed)),
        ood_max_splits=int(cfg.get("ood_max_splits", 10)),
        ood_targets=cfg.get("ood_targets"),
        ood_fractions=tuple(cfg.get("ood_fractions", [0.1, 0.2])),
        sparsex_neighbors=int(cfg.get("sparsex_neighbors", 5)),
        sparsey_center=str(cfg.get("sparsey_center", "median")),
        cv_seeds=cfg.get("cv_seeds"),
        period_strict=bool(cfg.get("ood_period_strict", False)),
        group_strict=bool(cfg.get("ood_group_strict", False)),
        output_dir=str(cfg.get("ood_output_dir", "./results/ood")),
    )


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
) -> None:
    """
    Entry point called from main.py when evaluation_mode == 'ood'.

    What it does:
    1) Loads train/test CSVs and concatenates them -> df_full
    2) Builds several OOD split families (LOEO / LOPO / LOGO / LOCO-k)
    3) For each split:
        - runs KFold on TRAIN portion only
        - evaluates on fixed held-out TEST portion
    4) Prints 4 tables + saves them as CSV (optional, but professional)
    """
    ood_cfg = _load_ood_cfg(cfg, default_seed=int(cv_random_state))

    # ---------- Load data ----------
    # 1) Build a FULL dataset

    if not train_dataset_path:
        raise ValueError("OOD mode requires train_dataset_path (can be the full dataset CSV).")

    # If train and test are the same file → use single dataset
    if (not test_dataset_path) or (
        os.path.normpath(test_dataset_path) == os.path.normpath(train_dataset_path)
    ):
        df_full = pd.read_csv(train_dataset_path).reset_index(drop=True)

    else:
        df_train = pd.read_csv(train_dataset_path).reset_index(drop=True)
        df_test = pd.read_csv(test_dataset_path).reset_index(drop=True)
        df_full = pd.concat([df_train, df_test], ignore_index=True)

    target_col = ood_cfg.target_column
    formula_col = ood_cfg.formula_column

    if target_col not in df_full.columns:
        raise ValueError(f"Missing target column '{target_col}' in train/test CSVs")
    if formula_col not in df_full.columns:
        raise ValueError(f"Missing formula column '{formula_col}' in train/test CSVs")

    # Features = everything except target and formula
    feature_cols = [c for c in df_full.columns if c not in (target_col, formula_col)]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after excluding target/formula columns.")

    X_full = df_full[feature_cols].copy()
    y_full = df_full[target_col].copy()

    # ---------- Chemistry helpers ----------
    element_to_group, element_to_period = load_periodic_table_map(pt_path)
    elements_per_row = extract_elements_series(df_full, formula_column=formula_col)

    # ---------- RF vs XGB names for significance table ----------
    rf_name = _name_for_key(model_registry, "rf") if "rf" in models else None
    xgb_name = _name_for_key(model_registry, "xgb") if "xgb" in models else None

    # ---------- Decide what to run ----------
    mode = ood_cfg.ood_mode
    supported_modes = {"element", "period", "group", "cluster", "cluster10", "sparsex", "sparsey", "all"}
    if mode not in supported_modes:
        raise ValueError(
            f"Invalid ood_mode '{mode}'. Supported values are: {sorted(supported_modes)}"
        )

    run_element = mode in {"element", "all"}
    run_period = mode in {"period", "all"}
    run_group = mode in {"group", "all"}
    run_cluster = mode in {"cluster", "cluster10", "all"}
    run_sparsex = mode in {"sparsex", "all"}
    run_sparsey = mode in {"sparsey", "all"}

    # ---------- Collect tables ----------
    all_t1: List[pd.DataFrame] = []
    all_t2: List[pd.DataFrame] = []
    all_t3: List[pd.DataFrame] = []

    seeds = list(ood_cfg.cv_seeds) if ood_cfg.cv_seeds is not None else [int(ood_cfg.ood_seed)]
    max_splits = int(ood_cfg.ood_max_splits)

    # If user supplies ood_targets, we use it for whichever mode is active.
    # NOTE: for period/group you should pass ints; we coerce to int list.
    user_targets = ood_cfg.ood_targets

    # ---------- LOEO ----------
    if run_element:
        if user_targets is None:
            targets = _top_elements(elements_per_row, max_splits)
        else:
            targets = [str(x) for x in user_targets]

        splits: List[Split] = build_loeo_splits(elements_per_row, targets)[:max_splits]

        for seed in seeds:
            t1, t2, t3 = evaluate_splits_kfold_train_fixed_test(
                X_full,
                y_full,
                splits,
                models,
                model_registry,
                scenario="LOEO",
                seed=int(seed),
                cv_folds=cv_folds,
                shuffle=cv_shuffle,
                hyperparameter_tuning=hyperparameter_tuning,
                rf_name=rf_name,
                xgb_name=xgb_name,
            )
            all_t1.append(t1)
            all_t2.append(t2)
            all_t3.append(t3)

    # ---------- LOPO ----------
    if run_period:
        if user_targets is None:
            targets = _top_periods(elements_per_row, element_to_period, max_splits)
        else:
            targets = _safe_int_list(user_targets) or []

        splits = build_period_splits(
            elements_per_row,
            element_to_period,
            targets,
            strict=bool(ood_cfg.period_strict),
        )[:max_splits]

        for seed in seeds:
            t1, t2, t3 = evaluate_splits_kfold_train_fixed_test(
                X_full,
                y_full,
                splits,
                models,
                model_registry,
                scenario="LOPO",
                seed=int(seed),
                cv_folds=cv_folds,
                shuffle=cv_shuffle,
                hyperparameter_tuning=hyperparameter_tuning,
                rf_name=rf_name,
                xgb_name=xgb_name,
            )
            all_t1.append(t1)
            all_t2.append(t2)
            all_t3.append(t3)

    # ---------- LOGO ----------
    if run_group:
        if user_targets is None:
            targets = _top_groups(elements_per_row, element_to_group, max_splits)
        else:
            targets = _safe_int_list(user_targets) or []

        splits = build_group_splits(
            elements_per_row,
            element_to_group,
            targets,
            strict=bool(ood_cfg.group_strict),
        )[:max_splits]

        for seed in seeds:
            t1, t2, t3 = evaluate_splits_kfold_train_fixed_test(
                X_full,
                y_full,
                splits,
                models,
                model_registry,
                scenario="LOGO",
                seed=int(seed),
                cv_folds=cv_folds,
                shuffle=cv_shuffle,
                hyperparameter_tuning=hyperparameter_tuning,
                rf_name=rf_name,
                xgb_name=xgb_name,
            )
            all_t1.append(t1)
            all_t2.append(t2)
            all_t3.append(t3)

    # ---------- LOCO-k ----------
    if run_cluster:
        k = int(ood_cfg.ood_k)
        splits = build_kmeans_cluster_splits(
            X_full,
            k=k,
            seed=int(ood_cfg.ood_seed),
        )[:max_splits]

        for seed in seeds:
            t1, t2, t3 = evaluate_splits_kfold_train_fixed_test(
                X_full,
                y_full,
                splits,
                models,
                model_registry,
                scenario=f"LOCO(k={k})",
                seed=int(seed),
                cv_folds=cv_folds,
                shuffle=cv_shuffle,
                hyperparameter_tuning=hyperparameter_tuning,
                rf_name=rf_name,
                xgb_name=xgb_name,
            )
            all_t1.append(t1)
            all_t2.append(t2)
            all_t3.append(t3)

    # ---------- SparseX ----------
    if run_sparsex:
        splits = build_sparsex_splits(
            X_full,
            fractions=ood_cfg.ood_fractions,
            n_neighbors=ood_cfg.sparsex_neighbors,
        )[:max_splits]

        for seed in seeds:
            t1, t2, t3 = evaluate_splits_kfold_train_fixed_test(
                X_full,
                y_full,
                splits,
                models,
                model_registry,
                scenario="SparseX",
                seed=int(seed),
                cv_folds=cv_folds,
                shuffle=cv_shuffle,
                hyperparameter_tuning=hyperparameter_tuning,
                rf_name=rf_name,
                xgb_name=xgb_name,
            )
            all_t1.append(t1)
            all_t2.append(t2)
            all_t3.append(t3)

    # ---------- SparseY ----------
    if run_sparsey:
        splits = build_sparsey_splits(
            y_full,
            fractions=ood_cfg.ood_fractions,
            center=ood_cfg.sparsey_center,
        )[:max_splits]

        for seed in seeds:
            t1, t2, t3 = evaluate_splits_kfold_train_fixed_test(
                X_full,
                y_full,
                splits,
                models,
                model_registry,
                scenario="SparseY",
                seed=int(seed),
                cv_folds=cv_folds,
                shuffle=cv_shuffle,
                hyperparameter_tuning=hyperparameter_tuning,
                rf_name=rf_name,
                xgb_name=xgb_name,
            )
            all_t1.append(t1)
            all_t2.append(t2)
            all_t3.append(t3)

    # ---------- Finalize / print ----------
    table1 = pd.concat(all_t1, ignore_index=True) if all_t1 else pd.DataFrame()
    table2 = pd.concat(all_t2, ignore_index=True) if all_t2 else pd.DataFrame()
    table3 = pd.concat(all_t3, ignore_index=True) if all_t3 else pd.DataFrame()
    table4 = summarize_runs_across_splits(table2) if not table2.empty else pd.DataFrame()

    print_ood_tables(table1, table2, table3, table4)

    # ---------- Save ----------
    out_dir = _ensure_dir(ood_cfg.output_dir)
    if not table1.empty:
        table1.to_csv(out_dir / "table1_splits_summary.csv", index=False)
    if not table2.empty:
        table2.to_csv(out_dir / "table2_metrics_by_model.csv", index=False)
    if not table3.empty:
        table3.to_csv(out_dir / "table3_rf_vs_xgb_significance.csv", index=False)
    if not table4.empty:
        table4.to_csv(out_dir / "table4_combined_comparison.csv", index=False)

    print(f"\nSaved OOD tables to: {out_dir.resolve()}")