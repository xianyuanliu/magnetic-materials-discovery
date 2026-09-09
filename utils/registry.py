"""What a model declares about itself, and how config-supplied model keys resolve against a registry of them.

Deliberately a leaf: it imports nothing from `loaddata`, `prepdata`, `pipeline`, `evaluate` or `interpret`, because
`evaluate/` needs `ModelSpec` too and must not import `pipeline/` to get it.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """One entry of a model registry.

    Attributes:
        key: Short identifier used in config files, e.g. "rf".
        name: Human-readable name used in result tables, e.g. "Random Forest".
        train: `(X, y, hyperparams=None, random_state=0) -> fitted estimator`.
        tune: `(X, y, cv_folds, random_state, n_iter) -> hyperparams dict`, or None when there is nothing to search.
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
