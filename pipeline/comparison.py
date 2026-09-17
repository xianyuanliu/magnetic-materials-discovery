"""Paired model comparison, shared by the holdout and cross_validation pipelines."""

from typing import Mapping, Tuple

from config import RunConfig
from utils.registry import ModelSpec, resolve_models
from evaluate.cross_validation import compare_models_significance
from utils.reporting import print_significance

# Only lower-is-better metrics: the paired tests read a positive difference as "model_a is worse", which r2 inverts.
COMPARISON_METRICS = ("mse", "mae")


def _comparison_names(cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> Tuple[str, str]:
    """Resolve `compare_models` to the two display names used in result tables."""
    specs = resolve_models(registry, cfg.compare_models)
    return specs[0].name, specs[1].name


def report_comparison(cfg: RunConfig, registry, results, test_train_ratio: float = 0.0) -> None:
    """Print the paired significance tests for the configured model pair.

    Args:
        cfg: The resolved run configuration, read for `compare_models`.
        registry: Registry to resolve those keys against.
        results: {model name: {metric: [one score per observation]}}.
        test_train_ratio: n_test / n_train for one observation, passed on so the t-test can correct for observations
            that share training data; see evaluate.cross_validation.compare_models_significance.
    """
    name_a, name_b = _comparison_names(cfg, registry)
    for metric in COMPARISON_METRICS:
        print_significance(compare_models_significance(
            results, name_a, name_b, metric=metric, test_train_ratio=test_train_ratio,
        ))
