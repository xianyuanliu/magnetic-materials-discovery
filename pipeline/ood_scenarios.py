"""Config-driven selection of OOD scenarios and their splits.

Turns a run config into a list of (scenario_name, splits) pairs by picking
which elements/periods/groups/clusters to hold out and delegating to
pipeline/ood_splits.py. Shared by the OOD and UQ pipelines so both evaluate
exactly the same scenarios.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from pipeline.ood_splits import (
    Split,
    build_group_splits,
    build_kmeans_cluster_splits,
    build_loeo_splits,
    build_period_splits,
    build_sparsex_splits,
    build_sparsey_splits,
)

SUPPORTED_MODES = ("element", "period", "group", "cluster", "sparsex", "sparsey", "all")

# Minimum split sizes. A handful of test samples makes R² meaningless and a
# tiny train set measures nothing but the sample count, so both are filtered
# out here rather than reported as OOD results.
DEFAULT_MIN_TEST = 10
DEFAULT_MIN_TRAIN = 50


@dataclass
class OODConfig:
    """Resolved OOD settings for one run (see load_ood_config)."""

    target_column: str = "saturation magnetization"
    formula_column: str = "chemical formula"

    # Which OOD scenarios to run; see SUPPORTED_MODES.
    ood_mode: str = "all"

    # LOCO settings.
    ood_k: int = 10

    # Selection and limits. ood_max_splits=None means "every split that passes
    # the minimum-size filters", which is the intended setting for a full run;
    # a number caps each scenario and is only useful for smoke tests.
    ood_seed: int = 0
    ood_max_splits: Optional[int] = None
    ood_targets: Optional[Sequence[Any]] = None  # elements OR periods OR groups
    ood_fractions: Sequence[float] = (0.1, 0.2)
    sparsex_neighbors: int = 5
    sparsey_center: str = "median"
    min_test: int = DEFAULT_MIN_TEST
    min_train: int = DEFAULT_MIN_TRAIN
    cv_seeds: Optional[Sequence[int]] = None

    # Strict membership for period/group (see _build_membership_splits).
    period_strict: bool = False
    group_strict: bool = False

    # Whether to score a same-size random control alongside each OOD split.
    size_matched_control: bool = True

    output_dir: str = "./results/ood"


def load_ood_config(cfg: dict, default_seed: int) -> OODConfig:
    """Build an OODConfig from the run config dict, filling in defaults."""
    max_splits = cfg.get("ood_max_splits")
    return OODConfig(
        target_column=str(cfg.get("target_column", "saturation magnetization")),
        formula_column=str(cfg.get("formula_column", "chemical formula")),
        ood_mode=str(cfg.get("ood_mode", "all")).lower(),
        ood_k=int(cfg.get("ood_k", 10)),
        ood_seed=int(cfg.get("ood_seed", default_seed)),
        ood_max_splits=None if max_splits is None else int(max_splits),
        ood_targets=cfg.get("ood_targets"),
        ood_fractions=tuple(cfg.get("ood_fractions", [0.1, 0.2])),
        sparsex_neighbors=int(cfg.get("sparsex_neighbors", 5)),
        sparsey_center=str(cfg.get("sparsey_center", "median")),
        min_test=int(cfg.get("ood_min_test", DEFAULT_MIN_TEST)),
        min_train=int(cfg.get("ood_min_train", DEFAULT_MIN_TRAIN)),
        cv_seeds=cfg.get("cv_seeds"),
        period_strict=bool(cfg.get("ood_period_strict", False)),
        group_strict=bool(cfg.get("ood_group_strict", False)),
        size_matched_control=bool(cfg.get("ood_size_matched_control", True)),
        output_dir=str(cfg.get("ood_output_dir", "./results/ood")),
    )


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

    Frequency order matters when a cap is applied: it puts the largest held-out
    sets first, which are also the splits with the least training data left, so
    a capped run is not a random sample of the scenario.
    """
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ordered = ordered if max_n is None else ordered[:max_n]
    return [key for key, _ in ordered]


def _capped(splits: List[Split], max_splits: Optional[int]) -> List[Split]:
    return splits if max_splits is None else splits[:max_splits]


def build_scenarios(
    ood_cfg: OODConfig,
    X: pd.DataFrame,
    y: pd.Series,
    elements_per_row: Sequence[Sequence[str]],
    element_to_group: Dict[str, int],
    element_to_period: Dict[str, int],
) -> List[Tuple[str, List[Split]]]:
    """Build every scenario selected by `ood_cfg.ood_mode`.

    Args:
        ood_cfg: Resolved settings.
        X: Feature matrix, used by the geometry-based families (LOCO, SparseX).
        y: Target, used by SparseY.
        elements_per_row: Parsed element symbols per sample.
        element_to_group: Element symbol to periodic-table group.
        element_to_period: Element symbol to periodic-table period.

    Returns:
        A list of (scenario_name, splits) pairs, skipping scenarios that end up
        with no split passing the minimum-size filters.

    Raises:
        ValueError: If ood_cfg.ood_mode is not one of SUPPORTED_MODES.
    """
    mode = ood_cfg.ood_mode
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Invalid ood_mode '{mode}'. Supported values are: {sorted(SUPPORTED_MODES)}"
        )

    sizes = dict(min_train=ood_cfg.min_train, min_test=ood_cfg.min_test)
    max_splits = ood_cfg.ood_max_splits
    user_targets = ood_cfg.ood_targets
    scenarios: List[Tuple[str, List[Split]]] = []

    def selected(name: str) -> bool:
        return mode in {name, "all"}

    if selected("element"):
        targets = (
            [str(t) for t in user_targets]
            if user_targets is not None
            else _ranked_targets(_counts_by_attr(elements_per_row), max_splits)
        )
        scenarios.append(("LOEO", build_loeo_splits(elements_per_row, targets, **sizes)))

    if selected("period"):
        targets = (
            [int(t) for t in user_targets]
            if user_targets is not None
            else _ranked_targets(_counts_by_attr(elements_per_row, element_to_period), max_splits)
        )
        scenarios.append((
            "LOPO",
            build_period_splits(
                elements_per_row, element_to_period, targets,
                strict=ood_cfg.period_strict, **sizes,
            ),
        ))

    if selected("group"):
        targets = (
            [int(t) for t in user_targets]
            if user_targets is not None
            else _ranked_targets(_counts_by_attr(elements_per_row, element_to_group), max_splits)
        )
        scenarios.append((
            "LOGO",
            build_group_splits(
                elements_per_row, element_to_group, targets,
                strict=ood_cfg.group_strict, **sizes,
            ),
        ))

    if selected("cluster"):
        k = ood_cfg.ood_k
        scenarios.append((
            f"LOCO(k={k})",
            build_kmeans_cluster_splits(X, k=k, seed=ood_cfg.ood_seed, **sizes),
        ))

    if selected("sparsex"):
        scenarios.append((
            "SparseX",
            build_sparsex_splits(
                X, fractions=ood_cfg.ood_fractions,
                n_neighbors=ood_cfg.sparsex_neighbors, **sizes,
            ),
        ))

    if selected("sparsey"):
        scenarios.append((
            "SparseY",
            build_sparsey_splits(
                y, fractions=ood_cfg.ood_fractions,
                center=ood_cfg.sparsey_center, **sizes,
            ),
        ))

    return [
        (name, _capped(splits, max_splits))
        for name, splits in scenarios
        if splits
    ]
