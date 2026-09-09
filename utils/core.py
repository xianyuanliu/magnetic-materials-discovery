"""The vocabulary every stage speaks: the split type, the metric list, and the model-registry entry.

Deliberately a leaf: it imports nothing from `loaddata`, `prepdata`, `pipeline`, `evaluate` or `interpret`. Keeping what
they all share here is what lets `evaluate/` score results without importing `pipeline/`.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

# A train/test split: an identifier plus the row positions on each side. Split builders live in pipeline/ood_splits.py;
# the evaluators only consume splits, so they annotate against this alias rather than importing that module.
Split = Tuple[str, np.ndarray, np.ndarray]

# Regression metrics carried through every result table, and the decimals each is printed with. MRE is a small
# fraction, so it needs more places.
METRICS = ("mse", "mae", "mre", "r2")
METRIC_DECIMALS = {"mse": 4, "mae": 4, "mre": 6, "r2": 4}


@dataclass(frozen=True)
class ModelSpec:
    """One entry of a model registry.

    Attributes:
        key: Short identifier used in config files, e.g. "rf".
        name: Human-readable name used in result tables, e.g. "Random Forest".
        train: `(X, y, params=None, random_state=0) -> fitted estimator`.
        tune: `(X, y, cv_folds, random_state, n_iter) -> params dict`, or None when the model has nothing to search.
    """

    key: str
    name: str
    train: Callable
    tune: Optional[Callable] = None


def resolve_models(registry: Mapping[str, ModelSpec], keys: Sequence[str]) -> Tuple[ModelSpec, ...]:
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
        raise ValueError(f"Unknown model key(s): {sorted(unknown)}. Available: {sorted(registry)}")
    return tuple(registry[key] for key in keys)
