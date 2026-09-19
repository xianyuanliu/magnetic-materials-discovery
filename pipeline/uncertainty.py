"""Uncertainty quantification: is the model's confidence earned?

For every split — an in-distribution K-fold reference, each shifted scenario, and a same-size random control per
shifted split — this fits one model, attaches three kinds of interval (see predict/uncertainty.py), and records one
row per predicted sample. Calibration is then computed by pooling those samples, so error and uncertainty are always
aggregated the same way.

What it deliberately does not do is score the "confidence-error paradox". The usual test for it — comparing a
shifted/in-distribution error ratio against the same ratio of uncertainties — flips sign with the random seed on this
data, so this module reports coverage and error/sigma against their nominal targets, with a spread across seeds, and
leaves the reading to the reader.

The model is chosen from ENSEMBLE_STD_MODELS rather than by hard-coding a Random Forest; see
resolve_uncertainty_model. Reads no file and takes no path: the caller brings the data and the scenarios.
"""

from __future__ import annotations

import warnings
import zlib
from typing import List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluate.calibration import summarize_across_seeds, summarize_calibration
from loaddata.splits import Split, build_kfold_splits, build_size_matched_split
from predict.uncertainty import (
    CONFORMAL,
    CONFORMAL_NORM,
    RF_STD,
    ConformalCalibrator,
    gaussian_half_width,
    rf_tree_std,
)
from utils.registry import ModelSpec
from utils.reporting import SILENT, Logger

# Split families, as they appear in the `split_type` column.
ID_KFOLD = "ID-kfold"
OOD = "OOD"
ID_RANDOM = "ID-random"

# Model keys whose fitted estimator exposes per-member predictions (sklearn's `estimators_`) that
# predict/uncertainty.py's rf_tree_std can read as a spread. RandomForestRegressor is a bag of interchangeable trees;
# XGBoost fits one additive model, so its boosters carry no comparable spread.
ENSEMBLE_STD_MODELS = frozenset({"rf"})

# Calibration metrics, as named in evaluate/calibration.py. These replace the regression metrics: an uncertainty run
# scores how well an interval is earned, not how close a point prediction is.
UNCERTAINTY_METRICS = ("mae", "coverage", "coverage_error", "mean_width", "rms_z")


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

    model = train_model(X.iloc[fit_idx], y.iloc[fit_idx], hyperparams=None, random_state=model_random_state)

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

    Each control keeps its parent's split_id so the two stay paired in the output tables.
    """
    controls = []
    for split_id, train_idx, test_idx in splits:
        control = build_size_matched_split(
            n_samples, len(train_idx), len(test_idx),
            # crc32, not hash(): str hashing is salted per process, and these seeds must be reproducible across runs.
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
                warnings.warn(f"Skipping {scenario} {split_id}: too few training rows to calibrate.")
                continue
            samples.insert(0, "n_train", len(train_idx))
            samples.insert(0, "split_id", split_id)
            samples.insert(0, "scenario", scenario)
            samples.insert(0, "split_type", split_type)
            samples.insert(0, "seed", seed)
            frames.append(samples)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def resolve_uncertainty_model(registry: Mapping[str, ModelSpec], model_key: str) -> ModelSpec:
    """Look up the uncertainty model and check it can actually supply a spread.

    The estimators in predict/uncertainty.py read the per-member predictions of an ensemble, so the model has to be in
    ENSEMBLE_STD_MODELS.

    Raises:
        ValueError: If the key is unknown, or names a model with no ensemble spread to read.
    """
    if model_key not in registry:
        raise ValueError(f"Unknown uncertainty_model {model_key!r}. Available: {sorted(registry)}")

    spec = registry[model_key]
    if model_key not in ENSEMBLE_STD_MODELS:
        usable = sorted(ENSEMBLE_STD_MODELS & set(registry))
        raise ValueError(
            f"uncertainty_model {model_key!r} ({spec.name}) exposes no ensemble spread, which the "
            f"uncertainty estimators need. Models that do: {usable}"
        )
    return spec


class UncertaintyTables(NamedTuple):
    """The three tables an uncertainty run produces.

    Attributes:
        by_split: Calibration per split, per interval method — the finest grain anything is reported at.
        pooled_by_seed: The same samples pooled over splits, one row per seed, split type and method.
        across_seeds: `pooled_by_seed` averaged over seeds, with the spread that says whether a gap survives reseeding.
    """

    by_split: pd.DataFrame
    pooled_by_seed: pd.DataFrame
    across_seeds: pd.DataFrame


def run_uncertainty_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    spec: ModelSpec,
    scenarios: Sequence[Tuple[str, List[Split]]],
    *,
    seeds: Sequence[int] = (0,),
    alpha: float = 0.05,
    calibration_fraction: float = 0.25,
    inner_folds: int = 5,
    inner_shuffle: bool = True,
    size_matched_control: bool = True,
    model_random_state: int = 0,
    logger: Logger = SILENT,
) -> Optional[UncertaintyTables]:
    """Score how well `spec`'s intervals hold up, in distribution and under every scenario given.

    Each split withholds `calibration_fraction` of its own training rows to calibrate the conformal intervals, so no
    interval is calibrated on rows it will be scored against.

    Args:
        X: Feature matrix covering every split's indices.
        y: Target.
        spec: The model to fit; it must expose an ensemble spread, which `resolve_uncertainty_model` checks.
        scenarios: (scenario_name, splits) pairs, typically from `pipeline.ood.build_scenarios`.
        seeds: One repeat per seed; a single seed moves coverage by more than the effects being compared.
        alpha: Nominal miscoverage, so 0.05 asks for 95% intervals.
        calibration_fraction: Fraction of each training set withheld to calibrate the conformal intervals.
        inner_folds, inner_shuffle: The in-distribution K-fold reference every scenario is read against.
        size_matched_control: Also score a random split of the same two sizes per shifted split.
        model_random_state: Seed for model construction.
        logger: Where progress goes while the run is still going.

    Returns:
        The three tables, or None when no split left enough rows to both fit and calibrate.
    """
    logger.write(
        f"\n[INFO] Uncertainty run: {spec.name}, {len(scenarios)} OOD scenario(s), "
        f"{sum(len(s) for _, s in scenarios)} split(s), {len(seeds)} seed(s), "
        f"{calibration_fraction:.0%} of each training set held out for conformal calibration.\n"
    )

    per_seed_frames = []
    for seed in seeds:
        scenario_splits: List[Tuple[str, str, List[Split]]] = [
            (ID_KFOLD, ID_KFOLD, build_kfold_splits(len(X), inner_folds, inner_shuffle, seed))
        ]
        for scenario, splits in scenarios:
            scenario_splits.append((OOD, scenario, splits))
            if size_matched_control:
                scenario_splits.append((ID_RANDOM, scenario, _controls_for(splits, len(X), seed, scenario)))

        per_seed_frames.append(_collect_samples(
            X, y, scenario_splits, spec.train,
            seed=seed, calibration_fraction=calibration_fraction,
            alpha=alpha, model_random_state=model_random_state,
        ))

    frames = [frame for frame in per_seed_frames if not frame.empty]
    if not frames:
        warnings.warn("No uncertainty samples produced; nothing to report.")
        return None
    samples = pd.concat(frames, ignore_index=True)

    by_split = summarize_calibration(samples, ("seed", "split_type", "scenario", "split_id", "method"), alpha)
    pooled_by_seed = summarize_calibration(samples, ("seed", "split_type", "method"), alpha)
    return UncertaintyTables(
        by_split=by_split,
        pooled_by_seed=pooled_by_seed,
        across_seeds=summarize_across_seeds(pooled_by_seed, ("split_type", "method")),
    )
