"""Shared types and defaults with no dependency on any pipeline stage.

This module is deliberately a leaf: it imports nothing from `loaddata`,
`prepdata`, `pipeline`, `evaluate` or `interpret`. Putting the vocabulary they
all share here is what lets `evaluate/` score results without importing
`pipeline/`, which previously made the two packages mutually dependent.
"""

import types
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

# A train/test split: an identifier plus the row positions on each side.
# Split builders live in pipeline/ood_splits.py; the evaluators only consume
# splits, so they annotate against this alias rather than importing that module.
Split = Tuple[str, np.ndarray, np.ndarray]

# Default hyperparameter-search budget. Deliberately smaller than the outer
# cv_folds: the search is nested inside every outer fold, so its cost is
# multiplied by (outer folds x seeds x models).
DEFAULT_TUNE_CV_FOLDS = 3
DEFAULT_TUNE_N_ITER = 20

# Regression metrics carried through every result table, and the decimals each
# is printed with. MRE is a small fraction, so it needs more places.
METRICS = ("mse", "mae", "mre", "r2")
METRIC_DECIMALS = {"mse": 4, "mae": 4, "mre": 6, "r2": 4}


@dataclass(frozen=True)
class ModelSpec:
    """One entry of a model registry.

    Replaces the previous `{"name": ..., "train": ..., "tune": ...}` dict, whose
    schema was enforced by nothing. A dataclass gives the registry a type, lets
    a model declare what it can do (rather than being recognised by key
    elsewhere in the codebase), and gives downstream projects something concrete
    to construct when they register their own models.

    Attributes:
        key: Short identifier used in config files, e.g. "rf".
        name: Human-readable name used in result tables, e.g. "Random Forest".
        train: `(X, y, params=None, random_state=0) -> fitted estimator`.
        tune: `(X, y, cv_folds, random_state, n_iter) -> params dict`, or None
            when the model has nothing to search.
        provides_ensemble_std: Whether a fitted estimator exposes per-member
            predictions (sklearn's `estimators_`) that the UQ pipeline can read
            as a spread. Declared here so pipeline/uq_pipeline.py can select a
            model by capability instead of hard-coding the key "rf".
    """

    key: str
    name: str
    train: Callable
    tune: Optional[Callable] = None
    provides_ensemble_std: bool = False


def build_registry(specs: Sequence[ModelSpec]) -> Mapping[str, ModelSpec]:
    """Index `specs` by key into a read-only registry.

    Args:
        specs: Model specifications, with unique keys.

    Returns:
        A mapping from key to ModelSpec that callers cannot mutate.

    Raises:
        ValueError: If two specs share a key.
    """
    registry: Dict[str, ModelSpec] = {}
    for spec in specs:
        if spec.key in registry:
            raise ValueError(f"Duplicate model key in registry: {spec.key!r}")
        registry[spec.key] = spec
    return types.MappingProxyType(registry)


def resolve_models(
    registry: Mapping[str, ModelSpec],
    keys: Sequence[str],
) -> Tuple[ModelSpec, ...]:
    """Look up every key in `registry`, failing fast on the first unknown one.

    Args:
        registry: Model registry to resolve against.
        keys: Model keys from a run config.

    Returns:
        The resolved specs, in the order the keys were given.

    Raises:
        ValueError: If a key is not in the registry, listing what is.
    """
    unknown = [key for key in keys if key not in registry]
    if unknown:
        raise ValueError(
            f"Unknown model key(s): {sorted(unknown)}. "
            f"Available: {sorted(registry)}"
        )
    return tuple(registry[key] for key in keys)
