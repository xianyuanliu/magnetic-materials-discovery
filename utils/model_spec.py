"""The model-registry vocabulary: what a model declares about itself, and how
config-supplied model keys resolve against it.

Deliberately a leaf, for the same reason as utils/core.py: `evaluate/` needs
`ModelSpec` too, and must not import `pipeline/` to get it.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

# Default hyperparameter-search budget. Deliberately smaller than the outer
# cv_folds: the search is nested inside every outer fold, so its cost is
# multiplied by (outer folds x seeds x models).
DEFAULT_TUNE_CV_FOLDS = 3
DEFAULT_TUNE_N_ITER = 20


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
    """

    key: str
    name: str
    train: Callable
    tune: Optional[Callable] = None


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
