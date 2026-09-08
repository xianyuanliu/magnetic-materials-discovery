"""Paired model comparison, shared by the holdout and cross_validation pipelines."""

from typing import Mapping, Tuple

from config import RunConfig
from core import ModelSpec, resolve_models
from evaluate.cross_validation import compare_models_significance, count_wins
from reporting import print_significance, print_win_counts

# Metrics the configured model pair is compared on.
COMPARISON_METRICS = ("mse", "mae")


def comparison_names(
    cfg: RunConfig, registry: Mapping[str, ModelSpec]
) -> Tuple[str, str]:
    """Resolve `compare_models` to the two display names used in result tables."""
    specs = resolve_models(registry, cfg.compare_models)
    return specs[0].name, specs[1].name


def report_comparison(cfg: RunConfig, registry, results, n_repeats: int) -> None:
    """Print the win tally and paired tests for the configured model pair."""
    name_a, name_b = comparison_names(cfg, registry)
    print_win_counts(count_wins(results, name_a, name_b, metric="mse"), n_repeats)
    for metric in COMPARISON_METRICS:
        print_significance(compare_models_significance(results, name_a, name_b, metric=metric))
