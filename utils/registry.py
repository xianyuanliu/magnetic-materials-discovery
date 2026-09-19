"""What a model declares about itself, and how config-supplied model keys resolve against a registry of them.

Deliberately a leaf: it imports nothing from `loaddata`, `prepdata`, `pipeline`, `evaluate` or `interpret`, because
`evaluate/` needs `ModelSpec` too and must not import `pipeline/` to get it.
"""

from dataclasses import dataclass
from typing import Callable, Collection, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """One entry of a model registry.

    Attributes:
        key: Short identifier used in config files, e.g. "rf".
        name: Human-readable name used in result tables, e.g. "Random Forest".
        train: `(X, y, hyperparams=None, random_state=0) -> fitted estimator`.
        tune: `(X, y, cv_folds, random_state, n_iter) -> hyperparams dict`, or None when there is nothing to
            search. The signature is uniform, so a model with nothing to seed or sample ignores those two.
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


def resolve_trained_models(
    registry: Mapping[str, ModelSpec],
    trained: Collection[str],
    keys: Sequence[str],
) -> Tuple[ModelSpec, ...]:
    """Look up every key, and check this run actually fitted the models it names.

    A config can name a model for a figure that the same config never put in `models`. Resolving alone would not
    catch that — the key is real, the registry has it, and only the lookup into the fitted models would fail, with a
    KeyError naming nothing useful.

    Args:
        registry: Model registry to resolve against.
        trained: The model keys this run fitted, typically the fitted-model mapping itself.
        keys: Model keys from a run config.

    Returns:
        The resolved specs, in the order the keys were given.

    Raises:
        ValueError: If a key is not in the registry, or names a model this run never fitted.
    """
    specs = resolve_models(registry, keys)
    missing = [spec.key for spec in specs if spec.key not in trained]
    if missing:
        raise ValueError(f"{missing} named for a figure, but this run fitted {sorted(trained)}")
    return specs
