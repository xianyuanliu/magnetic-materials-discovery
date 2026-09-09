"""Config-driven selection of OOD scenarios and their splits.

Turns a resolved OODConfig into a list of (scenario_name, splits) pairs by
picking which elements/periods/groups/clusters to hold out and delegating to
pipeline/ood_splits.py. Shared by the OOD and UQ pipelines so both evaluate
exactly the same scenarios.

Hold-out targets are named per family (`elements`, `periods`, `groups`) rather
than through one shared list: element targets are symbols and period/group
targets are integers, so a single list could not serve all three — under
`ood_mode: all` it failed with `invalid literal for int()`. Resolution and the
error message for the superseded key live in config.load_ood_config.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config import OODConfig
from utils.core import Split
from pipeline.ood_splits import (
    build_group_splits,
    build_kmeans_cluster_splits,
    build_loeo_splits,
    build_period_splits,
    build_sparsex_splits,
    build_sparsey_splits,
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
        ood_cfg: Resolved OOD settings; `ood_mode` is validated at parse time.
        X: Feature matrix, used by the geometry-based families (LOCO, SparseX).
        y: Target, used by SparseY.
        elements_per_row: Parsed element symbols per sample.
        element_to_group: Element symbol to periodic-table group.
        element_to_period: Element symbol to periodic-table period.

    Returns:
        A list of (scenario_name, splits) pairs, skipping scenarios that end up
        with no split passing the minimum-size filters.
    """
    sizes = dict(min_train=ood_cfg.min_train, min_test=ood_cfg.min_test)
    max_splits = ood_cfg.ood_max_splits
    scenarios: List[Tuple[str, List[Split]]] = []

    def selected(name: str) -> bool:
        return ood_cfg.ood_mode in {name, "all"}

    def targets_for(configured, element_to_attr=None) -> List[Any]:
        """Explicit targets from the config, else the most frequent ones."""
        if configured is not None:
            return list(configured)
        return _ranked_targets(_counts_by_attr(elements_per_row, element_to_attr), max_splits)

    if selected("element"):
        scenarios.append(("LOEO", build_loeo_splits(elements_per_row, targets_for(ood_cfg.elements), **sizes)))

    if selected("period"):
        scenarios.append((
            "LOPO",
            build_period_splits(
                elements_per_row, element_to_period,
                targets_for(ood_cfg.periods, element_to_period),
                strict=ood_cfg.period_strict, **sizes,
            ),
        ))

    if selected("group"):
        scenarios.append((
            "LOGO",
            build_group_splits(
                elements_per_row, element_to_group,
                targets_for(ood_cfg.groups, element_to_group),
                strict=ood_cfg.group_strict, **sizes,
            ),
        ))

    if selected("cluster"):
        k = ood_cfg.ood_k
        scenarios.append((f"LOCO(k={k})", build_kmeans_cluster_splits(X, k=k, seed=ood_cfg.ood_seed, **sizes)))

    if selected("sparsex"):
        scenarios.append((
            "SparseX",
            build_sparsex_splits(X, fractions=ood_cfg.fractions, n_neighbors=ood_cfg.sparsex_neighbors, **sizes),
        ))

    if selected("sparsey"):
        scenarios.append((
            "SparseY",
            build_sparsey_splits(y, fractions=ood_cfg.fractions, center=ood_cfg.sparsey_center, **sizes),
        ))

    return [(name, _capped(splits, max_splits)) for name, splits in scenarios if splits]
