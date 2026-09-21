"""Out-of-distribution evaluation: which splits exist, and how each one scores.

Two steps, both free of any run config. `build_scenarios` decides which split families to build and hands back
(scenario_name, splits) pairs, delegating the index arithmetic to loaddata/splits.py. `run_ood_evaluation` then scores
every split against its in-distribution references and returns the five result tables.

Which families *can* be built is a property of the data and lives in loaddata/splits.py; which ones a given run wants
is the caller's decision, expressed as arguments here.

Hold-out targets are named per family (`elements`, `periods`, `groups`) rather than through one shared list: element
targets are symbols and period/group targets are integers, so a single list could not serve all three.
"""

from __future__ import annotations

import warnings
import zlib
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluate.metrics import METRICS, FoldScores, build_result_rows, compute_metrics
from evaluate.ood_summaries import ID_PAIRED, ID_RANDOM, OOD
from predict.fitting import fit_models
from loaddata.splits import (
    Split,
    build_group_splits,
    build_kfold_splits,
    build_kmeans_cluster_splits,
    build_loeo_splits,
    build_period_splits,
    build_size_matched_split,
    build_sparsex_splits,
    build_sparsey_splits,
)
from prepdata.composition import get_elements_per_row, get_group_period_maps
from utils.registry import ModelSpec
from utils.reporting import SILENT, Logger


def _counts_by_attr(
    elements_per_row: Sequence[Sequence[str]],
    element_to_attr: Optional[Dict[str, int]] = None,
) -> Dict[Any, int]:
    """Count rows containing each element (or each element attribute)."""
    counts: Dict[Any, int] = {}
    for els in elements_per_row:
        if element_to_attr is None:
            keys = set(els)
        else:
            keys = {element_to_attr[e] for e in set(els) if e in element_to_attr}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _ranked_targets(counts: Dict[Any, int], max_n: Optional[int]) -> List[Any]:
    """Return targets most-frequent-first, capped at `max_n` when it is set.

    Frequency order matters when a cap is applied: it puts the largest held-out sets first, which are also the splits
    with the least training data left, so a capped run is not a random sample of the scenario.
    """
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ordered = ordered if max_n is None else ordered[:max_n]
    return [key for key, _ in ordered]


def _capped(splits: List[Split], max_splits: Optional[int]) -> List[Split]:
    return splits if max_splits is None else splits[:max_splits]


def resolve_split_elements(
    metadata: pd.DataFrame,
    formula_column: str,
    periodic_table: pd.DataFrame,
) -> Tuple[List[List[str]], Dict[str, int], Dict[str, int]]:
    """Turn the loaded formula column into everything the element, period and group families need.

    Args:
        metadata: Per-sample metadata from loaddata.tabular_access.load_feature_table.
        formula_column: Column in `metadata` holding the chemical formula.
        periodic_table: Periodic table data, already loaded; read for the group and period maps.

    Returns:
        (elements_per_row, element_to_group, element_to_period).

    Raises:
        ValueError: If `metadata` has no formula column, since every split family here is defined by chemistry.
    """
    if formula_column not in metadata.columns:
        raise ValueError(
            "Out-of-distribution and uncertainty splits hold out elements, periods and groups, so they need the "
            f"{formula_column!r} column, but the feature table has none. Rebuild it with build_feature_tables.py, "
            f"or point the run at a table that keeps the formula. Available metadata: {sorted(metadata.columns)}."
        )
    element_to_group, element_to_period = get_group_period_maps(periodic_table)
    return get_elements_per_row(metadata[formula_column]), element_to_group, element_to_period


def build_scenarios(
    X: pd.DataFrame,
    y: pd.Series,
    elements_per_row: Sequence[Sequence[str]],
    element_to_group: Dict[str, int],
    element_to_period: Dict[str, int],
    *,
    scenarios: Sequence[str] = ("element", "period", "group", "cluster", "sparsex", "sparsey"),
    min_train: int = 1,
    min_test: int = 1,
    max_splits: Optional[int] = None,
    elements: Optional[Sequence[str]] = None,
    periods: Optional[Sequence[int]] = None,
    groups: Optional[Sequence[int]] = None,
    period_strict: bool = False,
    group_strict: bool = False,
    n_clusters: int = 5,
    cluster_seed: int = 0,
    fractions: Sequence[float] = (0.1, 0.2),
    sparsex_neighbors: int = 5,
    sparsey_center: str = "median",
) -> List[Tuple[str, List[Split]]]:
    """Build the selected split families.

    Args:
        X: Feature matrix, used by the geometry-based families (LOCO, SparseX).
        y: Target, used by SparseY.
        elements_per_row: Parsed element symbols per sample.
        element_to_group: Element symbol to periodic-table group.
        element_to_period: Element symbol to periodic-table period.
        scenarios: Which to build. They are built in a fixed order regardless of the order given here.
        min_train, min_test: A split with fewer rows than these on either side is dropped rather than scored.
        max_splits: Cap on the splits kept per family, and on how many hold-out targets are chosen when none are
            named. Targets are taken most-frequent-first, so a capped run keeps the largest held-out sets rather
            than an arbitrary sample.
        elements, periods, groups: Explicit hold-out targets per family; None picks the most frequent ones.
        period_strict, group_strict: Hold out a sample if *every* element belongs to the period/group, rather than any.
        n_clusters, cluster_seed: k and the seed for the k-means cluster family (LOCO).
        fractions: Held-out fraction per split for the two sparsity families.
        sparsex_neighbors: Neighbours used to measure how isolated a sample is in feature space.
        sparsey_center: "median" or "mean", the point target values are ranked away from.

    Returns:
        A list of (scenario_name, splits) pairs, skipping scenarios that end up with no split passing the
        minimum-size filters.
    """
    sizes = dict(min_train=min_train, min_test=min_test)
    built: List[Tuple[str, List[Split]]] = []

    def selected(name: str) -> bool:
        return name in scenarios

    def targets_for(configured, element_to_attr=None) -> List[Any]:
        """Explicit targets from the config, else the most frequent ones."""
        if configured is not None:
            return list(configured)
        return _ranked_targets(_counts_by_attr(elements_per_row, element_to_attr), max_splits)

    if selected("element"):
        built.append(("LOEO", build_loeo_splits(elements_per_row, targets_for(elements), **sizes)))

    if selected("period"):
        built.append((
            "LOPO",
            build_period_splits(
                elements_per_row, element_to_period,
                targets_for(periods, element_to_period),
                strict=period_strict, **sizes,
            ),
        ))

    if selected("group"):
        built.append((
            "LOGO",
            build_group_splits(
                elements_per_row, element_to_group,
                targets_for(groups, element_to_group),
                strict=group_strict, **sizes,
            ),
        ))

    if selected("cluster"):
        built.append((
            f"LOCO(k={n_clusters})",
            build_kmeans_cluster_splits(X, k=n_clusters, seed=cluster_seed, **sizes),
        ))

    if selected("sparsex"):
        built.append((
            "SparseX",
            build_sparsex_splits(X, fractions=fractions, n_neighbors=sparsex_neighbors, **sizes),
        ))

    if selected("sparsey"):
        built.append((
            "SparseY",
            build_sparsey_splits(y, fractions=fractions, center=sparsey_center, **sizes),
        ))

    return [(name, _capped(splits, max_splits)) for name, splits in built if splits]


def build_size_matched_controls(
    splits: Sequence[Split],
    n_samples: int,
    seed: int,
    scenario: str,
) -> Dict[str, Split]:
    """One same-size random split per shifted split, keyed by the parent's split_id.

    The control holds the shift constant against everything else a split changes: it draws the same number of training
    rows and the same number of test rows, at random. The shifted score minus its control is therefore attributable to
    the shift rather than to the smaller training set the shift left behind.

    Returns:
        {split_id: control split}, omitting splits whose sizes do not fit in `n_samples`.
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


def _empty_scores(specs: Sequence[ModelSpec]) -> FoldScores:
    return {spec.name: {metric: [] for metric in METRICS} for spec in specs}


def _mean_over_folds(scores: FoldScores) -> FoldScores:
    """Collapse each metric's per-fold values to their mean, as one observation per split.

    The inner folds all score the same fixed test set, so their spread is sensitivity to the training subsample, not
    test-set uncertainty; one number per split is what the paired comparisons need.
    """
    return {model: {metric: [float(np.mean(values))] for metric, values in per_metric.items()}
            for model, per_metric in scores.items()}


def heldout_label(split_id: str) -> str:
    """Extract what a split holds out from its id, or "" when the split holds out no named thing.

    Membership splits are named `E=Fe`, `P=4`, `G=8`, `C=3`; the sparsity families are named `SparseX_top10pct` and
    hold out a *fraction*, not a target, so they have no label to extract.
    """
    return split_id.split("=", 1)[1] if "=" in split_id else ""


def _score_fold_models(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    specs: Sequence[ModelSpec],
    *,
    inner_splits: Sequence[Split],
    score_inner: bool,
    hyperparameter_tuning: bool,
    model_random_state: int,
    tune_cv_folds: int,
    tune_n_iter: int,
) -> Tuple[FoldScores, Optional[FoldScores]]:
    """Fit one model per inner fold and score it on the split's fixed test set.

    The inner splits exist to average over training subsamples, not to select anything: no fold's score feeds back
    into fitting.

    Args:
        score_inner: Also score every fold model on the inner validation fold it held out. Free — the models are
            already fitted — and it is the only reference trained on exactly the same rows as the OOD score.

    Returns:
        (test_scores, inner_scores); inner_scores is None when score_inner is False.
    """
    X_train_full, y_train_full = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    test_scores = _empty_scores(specs)
    inner_scores = _empty_scores(specs) if score_inner else None

    for _, fit_idx, inner_idx in inner_splits:
        X_fit, y_fit = X_train_full.iloc[fit_idx], y_train_full.iloc[fit_idx]
        X_inner, y_inner = X_train_full.iloc[inner_idx], y_train_full.iloc[inner_idx]

        fitted = fit_models(
            X_fit, y_fit, specs,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )

        for spec in specs:
            model = fitted[spec.key]
            for metric, value in compute_metrics(y_test, model.predict(X_test)).items():
                test_scores[spec.name][metric].append(value)
            if score_inner:
                for metric, value in compute_metrics(y_inner, model.predict(X_inner)).items():
                    inner_scores[spec.name][metric].append(value)

    return test_scores, inner_scores


def score_splits_with_inner_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Split],
    specs: Sequence[ModelSpec],
    *,
    scenario: str,
    seed: int,
    inner_folds: int,
    inner_shuffle: bool,
    hyperparameter_tuning: bool,
    model_random_state: int = 0,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
    controls: Optional[Mapping[str, Split]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every split of one scenario for one seed, each beside its in-distribution references.

    Args:
        X, y: The whole pool.
        splits: The shifted splits to score.
        specs: Resolved model specifications.
        scenario: Name carried into the output rows.
        seed: Seeds the inner folds; carried into the output rows.
        inner_folds, inner_shuffle: The inner K-fold over each split's training rows.
        hyperparameter_tuning, tune_cv_folds, tune_n_iter: The search, as in `fit_models`.
        model_random_state: Seed for model construction and the search.
        controls: Size-matched random split per split_id, scored under ID_RANDOM when present.

    Returns:
        (splits, results): one row per split describing what it held out and how many rows ended up on each side,
        and the long-form result rows — one value per (split, split_type, model, metric), the mean over inner folds.
    """
    split_rows: List[Dict] = []
    result_frames: List[pd.DataFrame] = []

    def rows(split_id: str, split_type: str, scores: FoldScores) -> pd.DataFrame:
        return build_result_rows(
            _mean_over_folds(scores), scenario=scenario, split_type=split_type, split_ids=[split_id], seed=seed,
        )

    for split_id, train_idx, test_idx in splits:
        inner_splits = build_kfold_splits(len(train_idx), inner_folds, inner_shuffle, seed)
        if not inner_splits:
            warnings.warn(f"Skipping split {split_id} in {scenario}: n_train={len(train_idx)} leaves no inner folds")
            continue

        score = partial(
            _score_fold_models,
            X, y, specs=specs,
            inner_splits=inner_splits,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds, tune_n_iter=tune_n_iter,
        )

        ood_scores, id_paired_scores = score(train_idx, test_idx, score_inner=True)
        split_rows.append(dict(
            scenario=scenario, split_id=split_id, heldout_label=heldout_label(split_id),
            n_train=len(train_idx), n_test=len(test_idx), seed=seed,
        ))
        result_frames.append(rows(split_id, OOD, ood_scores))
        result_frames.append(rows(split_id, ID_PAIRED, id_paired_scores))

        control = (controls or {}).get(split_id)
        if control is not None:
            _, control_train_idx, control_test_idx = control
            control_scores, _ = score(control_train_idx, control_test_idx, score_inner=False)
            result_frames.append(rows(split_id, ID_RANDOM, control_scores))

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    return pd.DataFrame(split_rows), results


def run_ood_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    specs: Sequence[ModelSpec],
    scenarios: Sequence[Tuple[str, List[Split]]],
    *,
    seeds: Sequence[int] = (0,),
    inner_folds: int = 5,
    inner_shuffle: bool = True,
    size_matched_control: bool = True,
    hyperparameter_tuning: bool = False,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
    model_random_state: int = 0,
    logger: Logger = SILENT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every scenario's splits against their in-distribution references.

    Each split is fitted on its own training rows through an inner K-fold. Each is scored twice more: on the inner
    validation fold it already held out, and — when `size_matched_control` is on — on a random split of the same two
    sizes. That is what turns "the model got worse" into "the model got worse *because of the shift*".

    Args:
        X: Feature matrix covering every split's indices.
        y: Target.
        specs: Resolved model specifications to score.
        scenarios: (scenario_name, splits) pairs, typically from `build_scenarios`.
        seeds: One repeat per seed, seeding the inner folds and the controls.
        inner_folds, inner_shuffle: The inner K-fold over each split's training rows.
        size_matched_control: Also score a random split of the same two sizes per shifted split.
        hyperparameter_tuning: Search inside every inner fold.
        tune_cv_folds, tune_n_iter: That search's budget.
        model_random_state: Seed for model construction and the search.
        logger: Where progress goes while the run is still going.

    Returns:
        (splits, results): what every split held out, and the long-form result rows for every scenario and seed.

    Raises:
        ValueError: If `scenarios` is empty, since the run would otherwise write empty tables and exit cleanly.
    """
    if not scenarios:
        raise ValueError("No scenario has a split; nothing to score.")

    logger.write(
        f"\n[INFO] OOD run: {len(scenarios)} scenario(s), "
        f"{sum(len(s) for _, s in scenarios)} split(s), {len(seeds)} seed(s), "
        f"size-matched control {'on' if size_matched_control else 'off'}.\n"
    )
    for name, splits in scenarios:
        logger.write(f"         {name}: {len(splits)} split(s) — {[s[0] for s in splits]}\n")

    split_frames: List[pd.DataFrame] = []
    result_frames: List[pd.DataFrame] = []
    for scenario, splits in scenarios:
        for seed in seeds:
            controls = (
                build_size_matched_controls(splits, len(X), int(seed), scenario) if size_matched_control else None
            )
            split_frame, result_frame = score_splits_with_inner_kfold(
                X, y, splits, specs,
                scenario=scenario, seed=int(seed),
                inner_folds=inner_folds, inner_shuffle=inner_shuffle,
                hyperparameter_tuning=hyperparameter_tuning,
                model_random_state=model_random_state,
                tune_cv_folds=tune_cv_folds, tune_n_iter=tune_n_iter,
                controls=controls,
            )
            split_frames.append(split_frame)
            result_frames.append(result_frame)

    return pd.concat(split_frames, ignore_index=True), pd.concat(result_frames, ignore_index=True)
