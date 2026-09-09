"""Uncertainty-quantification orchestration: is the model's confidence earned?

For every split — an in-distribution K-fold reference, each OOD scenario from pipeline/ood_scenarios.py, and a same-size
random control per OOD split — this fits one model, attaches three kinds of interval (see pipeline/uq.py), and records
one row per predicted sample. Calibration is then computed by pooling those samples, so error and uncertainty are always
aggregated the same way.

What it deliberately does not do is score the "confidence-error paradox". The usual test for it — comparing an OOD/ID
error ratio against an OOD/ID uncertainty ratio — flips sign with the random seed on this data, so this pipeline reports
coverage and error/sigma against their nominal targets, with a spread across seeds, and leaves the reading to the
reader.

The model is chosen from ENSEMBLE_STD_MODELS rather than by hard-coding a Random Forest; see resolve_uq_model.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import RunConfig
from evaluate.ood_evaluation import Split
from utils.registry import ModelSpec
from evaluate.calibration import summarize_across_seeds, summarize_calibration
from pipeline.ood_pipeline import load_ood_dataset
from pipeline.ood_scenarios import build_scenarios
from pipeline.ood_splits import build_kfold_splits, build_size_matched_split
from pipeline.uq import (
    CONFORMAL,
    CONFORMAL_NORM,
    RF_STD,
    ConformalCalibrator,
    gaussian_half_width,
    rf_tree_std,
)
from prepdata.alloy_transform import extract_elements_series, load_periodic_table_map
from utils.reporting import print_uq_report

# Split families, as they appear in the `split_type` column.
ID_KFOLD = "ID-kfold"
OOD = "OOD"
ID_RANDOM = "ID-random"

# Model keys whose fitted estimator exposes per-member predictions (sklearn's `estimators_`) that pipeline/uq.py's
# rf_tree_std can read as a spread. RandomForestRegressor is a bag of interchangeable trees; XGBoost fits one additive
# model, so its boosters carry no comparable spread.
ENSEMBLE_STD_MODELS = frozenset({"rf"})

TABLE_FILENAMES = (
    "uq_table1_by_split.csv",
    "uq_table2_pooled_by_seed.csv",
    "uq_table3_across_seeds.csv",
)


def _fit_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_model,
    *,
    calibration_fraction: float,
    alpha: float,
    seed: int,
    model_random_state: int,
) -> Optional[pd.DataFrame]:
    """Fit on part of `train_idx`, calibrate on the rest, predict `test_idx`.

    Returns:
        One row per test sample and uncertainty method, or None if the split
        leaves too few rows to both fit and calibrate.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(train_idx))
    n_calibration = int(round(len(shuffled) * calibration_fraction))
    if n_calibration < 2 or len(shuffled) - n_calibration < 2:
        return None

    cal_idx, fit_idx = shuffled[:n_calibration], shuffled[n_calibration:]

    model = train_model(X.iloc[fit_idx], y.iloc[fit_idx], params=None, random_state=model_random_state)

    sigma_cal = rf_tree_std(model, X.iloc[cal_idx])
    y_cal = y.iloc[cal_idx].to_numpy(dtype=float)
    pred_cal = model.predict(X.iloc[cal_idx])

    sigma = rf_tree_std(model, X.iloc[test_idx])
    y_true = y.iloc[test_idx].to_numpy(dtype=float)
    y_pred = model.predict(X.iloc[test_idx])

    half_widths = {
        RF_STD: gaussian_half_width(sigma, alpha),
        CONFORMAL: ConformalCalibrator.fit(y_cal, pred_cal, alpha).half_width(sigma),
        CONFORMAL_NORM: ConformalCalibrator.fit(y_cal, pred_cal, alpha, sigma_cal).half_width(sigma),
    }

    return pd.concat([
        pd.DataFrame({"method": method, "y_true": y_true, "y_pred": y_pred, "sigma": sigma, "half_width": half_width})
        for method, half_width in half_widths.items()
    ], ignore_index=True)


def _controls_for(splits: Sequence[Split], n_samples: int, seed: int, scenario: str) -> List[Split]:
    """One same-size random split per OOD split, to separate shift from data loss.

    Each control keeps its parent's split_id so the two stay paired in the output tables. crc32, not hash(): str hashing
    is salted per process, and these seeds have to be reproducible across runs.
    """
    controls = []
    for split_id, train_idx, test_idx in splits:
        control = build_size_matched_split(
            n_samples, len(train_idx), len(test_idx),
            seed=zlib.crc32(f"{seed}|{scenario}|{split_id}".encode()),
            split_id=split_id,
        )
        if control is not None:
            controls.append(control)
    return controls


def _collect_samples(
    X: pd.DataFrame,
    y: pd.Series,
    scenario_splits: Sequence[Tuple[str, str, List[Split]]],
    train_model,
    *,
    seed: int,
    calibration_fraction: float,
    alpha: float,
    model_random_state: int,
) -> pd.DataFrame:
    """Run every split of every scenario for one seed and stack the sample rows."""
    frames = []
    for split_type, scenario, splits in scenario_splits:
        for split_id, train_idx, test_idx in splits:
            samples = _fit_and_predict(
                X, y, train_idx, test_idx, train_model,
                calibration_fraction=calibration_fraction, alpha=alpha,
                seed=seed, model_random_state=model_random_state,
            )
            if samples is None:
                print(f"[WARN] Skipping {scenario} {split_id}: too few training rows to calibrate")
                continue
            samples.insert(0, "n_train", len(train_idx))
            samples.insert(0, "split_id", split_id)
            samples.insert(0, "scenario", scenario)
            samples.insert(0, "split_type", split_type)
            samples.insert(0, "seed", seed)
            frames.append(samples)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def resolve_uq_model(model_registry: Mapping[str, ModelSpec], model_key: str) -> ModelSpec:
    """Look up the UQ model and check it can actually supply a spread.

    The estimators in pipeline/uq.py read the per-member predictions of an ensemble, so the model has to be in
    ENSEMBLE_STD_MODELS.

    Raises:
        ValueError: If the key is unknown, or names a model with no ensemble spread to read.
    """
    if model_key not in model_registry:
        raise ValueError(f"Unknown uq_model {model_key!r}. Available: {sorted(model_registry)}")

    spec = model_registry[model_key]
    if model_key not in ENSEMBLE_STD_MODELS:
        usable = sorted(ENSEMBLE_STD_MODELS & set(model_registry))
        raise ValueError(
            f"uq_model {model_key!r} ({spec.name}) exposes no ensemble spread, which the "
            f"uncertainty estimators need. Models that do: {usable}"
        )
    return spec


def run_uq_evaluation(*, cfg: RunConfig, model_registry: Mapping[str, ModelSpec]) -> None:
    """Run the UQ pipeline: score ID, OOD and control splits, then save the tables.

    Called from main.py when evaluation_mode == 'uq'.

    Args:
        cfg: The resolved run configuration.
        model_registry: Registry to resolve `uq_model` against.
    """
    uq_cfg = cfg.uq
    ood_cfg = cfg.ood
    spec = resolve_uq_model(model_registry, uq_cfg.model)
    alpha = uq_cfg.alpha
    calibration_fraction = uq_cfg.calibration_fraction
    seeds = list(uq_cfg.seeds)

    X, y, df_full = load_ood_dataset(
        cfg.dataset_path, cfg.dataset_path,
        cfg.target_column, cfg.formula_column, cfg.feature_columns,
    )
    element_to_group, element_to_period = load_periodic_table_map(cfg.pt_path)
    elements_per_row = extract_elements_series(df_full, formula_column=cfg.formula_column)

    ood_scenarios = build_scenarios(ood_cfg, X, y, elements_per_row, element_to_group, element_to_period)

    print(
        f"\n[INFO] UQ run: {spec.name}, {len(ood_scenarios)} OOD scenario(s), "
        f"{sum(len(s) for _, s in ood_scenarios)} split(s), {len(seeds)} seed(s), "
        f"{calibration_fraction:.0%} of each training set held out for conformal calibration."
    )

    per_seed_frames = []
    for seed in seeds:
        scenario_splits: List[Tuple[str, str, List[Split]]] = [
            (ID_KFOLD, ID_KFOLD, build_kfold_splits(len(X), cfg.kfold.folds, cfg.kfold.shuffle, seed))
        ]
        for scenario, splits in ood_scenarios:
            scenario_splits.append((OOD, scenario, splits))
            if ood_cfg.size_matched_control:
                scenario_splits.append((ID_RANDOM, scenario, _controls_for(splits, len(X), seed, scenario)))

        per_seed_frames.append(_collect_samples(
            X, y, scenario_splits, spec.train,
            seed=seed, calibration_fraction=calibration_fraction,
            alpha=alpha, model_random_state=cfg.model_random_state,
        ))

    frames = [f for f in per_seed_frames if not f.empty]
    if not frames:
        print("[WARN] No UQ samples produced; nothing to report.")
        return
    samples = pd.concat(frames, ignore_index=True)

    by_split = summarize_calibration(samples, ("seed", "split_type", "scenario", "split_id", "method"), alpha)
    pooled_by_seed = summarize_calibration(samples, ("seed", "split_type", "method"), alpha)
    across_seeds = summarize_across_seeds(pooled_by_seed, ("split_type", "method"))

    print_uq_report(across_seeds, alpha)

    out_dir = Path(uq_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for table, filename in zip((by_split, pooled_by_seed, across_seeds), TABLE_FILENAMES):
        table.to_csv(out_dir / filename, index=False)

    print(f"\nSaved UQ tables to: {out_dir.resolve()}")
