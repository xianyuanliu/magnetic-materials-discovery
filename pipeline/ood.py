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
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd

from evaluate.ood_evaluation import (
    OOD,
    score_splits_with_inner_kfold,
    summarize_generalization_gap,
    summarize_model_comparison,
    summarize_runs_across_splits,
)
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


def _score_scenario(
    scenario: str,
    splits: List[Split],
    seeds: Sequence[int],
    n_samples: int,
    size_matched_control: bool,
    score,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Score one scenario's splits across all seeds.

    Returns:
        (splits_summary_frames, metrics_frames) — one pair of tables per seed.
    """
    frames: Tuple[List, List] = ([], [])
    for seed in seeds:
        controls = (
            build_size_matched_controls(splits, n_samples, int(seed), scenario)
            if size_matched_control else None
        )
        tables = score(splits, scenario=scenario, seed=int(seed), controls=controls)
        for collected, table in zip(frames, tables):
            collected.append(table)
    return frames


def _warn_on_skip(scenario: str):
    """Return an on_skip callback that names the scenario a skipped split came from."""
    def _on_skip(split_id: str, reason: str) -> None:
        warnings.warn(f"Skipping split {split_id} in {scenario}: {reason}")

    return _on_skip


class OODTables(NamedTuple):
    """The five tables an out-of-distribution run produces.

    Attributes:
        splits_summary: One row per split — what it held out and how many rows ended up on each side.
        metrics_by_model: Per split, per model, per split type; the mean and standard deviation over inner folds.
        comparison_significance: Paired comparison of two models across shifted splits, empty when none was asked for.
        combined_comparison: The same metrics pooled across splits, one row per scenario and split type.
        generalization_gap: The shifted drop decomposed into a test-side shift and a smaller-training-pool part.
    """

    splits_summary: pd.DataFrame
    metrics_by_model: pd.DataFrame
    comparison_significance: pd.DataFrame
    combined_comparison: pd.DataFrame
    generalization_gap: pd.DataFrame


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
    compare: Optional[Tuple[str, str]] = None,
    logger: Logger = SILENT,
) -> OODTables:
    """Score every scenario's splits against their in-distribution references.

    Each split is fitted on its own training rows through an inner K-fold, so the spread reported per split measures
    sensitivity to the training subsample rather than test-set noise. Each is scored twice more: on the inner
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
        compare: Two model *names* to significance-test across shifted splits, or None to skip.
        logger: Where progress goes while the run is still going.

    Returns:
        The five result tables, as an `OODTables`.

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

    collected: Tuple[List, List] = ([], [])
    for scenario, splits in scenarios:
        score = partial(
            score_splits_with_inner_kfold,
            X, y, specs=specs,
            inner_splitter=lambda n_train, seed: build_kfold_splits(n_train, inner_folds, inner_shuffle, seed),
            hyperparameter_tuning=hyperparameter_tuning, best_hyperparams=None,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds, tune_n_iter=tune_n_iter,
            on_skip=_warn_on_skip(scenario),
        )
        scenario_frames = _score_scenario(scenario, splits, seeds, len(X), size_matched_control, score)
        for target, frames in zip(collected, scenario_frames):
            target.extend(frames)

    splits_summary, metrics_by_model = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for frames in collected
    )

    comparison_significance = pd.DataFrame()
    if compare is not None:
        comparison_significance = summarize_model_comparison(metrics_by_model, *compare, split_type=OOD)

    combined_comparison = summarize_runs_across_splits(metrics_by_model)
    return OODTables(
        splits_summary=splits_summary,
        metrics_by_model=metrics_by_model,
        comparison_significance=comparison_significance,
        combined_comparison=combined_comparison,
        generalization_gap=summarize_generalization_gap(combined_comparison),
    )
