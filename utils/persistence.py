"""Saving and loading what a run produces: fitted models, and the tables of numbers they scored.

A fitted estimator on its own is not enough to make a prediction: the caller also has to know which feature columns it
was trained on, in which order, and what the values mean. Bundling those with the estimator is what lets a model outlive
the run that produced it, which is the difference between a pipeline that reports scores and a library something else
can build on. The result tables are the same idea for the numbers — every figure and significance test in the repo is
derived from them, so a run that saves them can be re-analysed without being repeated.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import joblib
import pandas as pd

from evaluate.metrics import SUMMARY_GROUPS, summarize_scores

# Bumped when the bundle layout changes in a way older readers cannot handle.
BUNDLE_FORMAT_VERSION = 1


@dataclass
class ModelBundle:
    """A fitted model plus everything needed to apply it to new compositions.

    Attributes:
        model: The fitted estimator, with a `predict` method.
        model_key: Registry key the model was built from, e.g. "rf".
        model_name: Human-readable model name.
        feature_columns: Feature names in training order. `predict` must be given these columns, in this order.
        target_column: Name of the property the model predicts.
        dataset: Name of the dataset it was trained on.
        n_train: Number of training rows.
        metadata: Anything else worth recording, e.g. the fitted parameters.
        format_version: Layout version of the saved file.
    """

    model: Any
    model_key: str
    model_name: str
    feature_columns: Tuple[str, ...]
    target_column: str
    dataset: str = ""
    n_train: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    format_version: int = BUNDLE_FORMAT_VERSION

    def describe(self) -> str:
        """One-line summary for console output."""
        return (
            f"{self.model_name} ({self.model_key}) trained on {self.n_train} row(s) "
            f"of {self.dataset or 'an unnamed dataset'}, "
            f"predicting {self.target_column!r} from {len(self.feature_columns)} feature(s)"
        )


def save_model_bundle(path: str, bundle: ModelBundle) -> Path:
    """Write `bundle` to `path`, creating parent directories as needed.

    Args:
        path: Destination file, conventionally with a `.joblib` suffix.
        bundle: The bundle to persist.

    Returns:
        The resolved path written to.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination.resolve()


def load_model_bundle(path: str) -> ModelBundle:
    """Read a bundle written by save_model_bundle.

    Args:
        path: File to read.

    Returns:
        The loaded bundle.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file does not hold a ModelBundle, or holds one written by a newer, incompatible layout.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No saved model at {source}")

    bundle = joblib.load(source)
    if not isinstance(bundle, ModelBundle):
        raise ValueError(f"{source} does not contain a ModelBundle (got {type(bundle).__name__}).")
    if bundle.format_version > BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"{source} was written in bundle format v{bundle.format_version}, but this "
            f"version of the code understands at most v{BUNDLE_FORMAT_VERSION}."
        )
    return bundle


def align_features(features, feature_columns: Sequence[str], source: Optional[str] = None):
    """Select `feature_columns` from `features`, in training order.

    Column *order* matters to a fitted estimator, and a frame built by a different code path is under no obligation to
    preserve it — so the selection is explicit rather than assumed.

    Args:
        features: Frame containing at least the named columns.
        feature_columns: Names in training order.
        source: Optional label used in the error message.

    Returns:
        A frame with exactly `feature_columns`, in order.

    Raises:
        ValueError: If any expected column is missing.
    """
    missing = [c for c in feature_columns if c not in features.columns]
    if missing:
        where = f" in {source}" if source else ""
        raise ValueError(f"Missing feature column(s){where}: {missing}. The model expects {list(feature_columns)}.")
    return features[list(feature_columns)]


RESULTS_FILENAME = "results.csv"
SUMMARY_FILENAME = "summary.csv"


def save_tables(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Optional[str],
    *,
    enabled: bool = True,
) -> Optional[Path]:
    """Write a mode's derived tables, honoring the same switch that governs the raw results.

    Derived tables are analyses of the raw table — a gap decomposition, a calibration summary — so they are worth
    keeping together with it and worth skipping together with it.

    Args:
        tables: {filename: table}. An empty table is skipped, since a file of headers reads as a result of zero.
        output_dir: Directory to write into; saving is skipped when it is empty.
        enabled: False writes nothing.

    Returns:
        The directory written to, or None when nothing was written.
    """
    if not enabled or not output_dir:
        return None

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        if not table.empty:
            table.to_csv(directory / filename, index=False)
    return directory


def save_results(
    results: pd.DataFrame,
    output_dir: Optional[str],
    *,
    prefix: str = "",
    by: Sequence[str] = SUMMARY_GROUPS,
    enabled: bool = True,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """Summarize a long-form result table, and write both tables unless saving is turned off.

    Args:
        results: Rows shaped like evaluate.metrics.RESULT_COLUMNS.
        output_dir: Directory to write into; saving is skipped when it is empty.
        prefix: Prepended to both filenames, so several runs can share a directory.
        by: Identifying columns the summary groups on.
        enabled: False computes the summary but writes nothing, for a caller that only wants the numbers.

    Returns:
        (summary, directory), where directory is None when nothing was written.
    """
    summary = summarize_scores(results, by=by)

    if not enabled or not output_dir:
        return summary, None

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(directory / f"{prefix}{RESULTS_FILENAME}", index=False)
    summary.to_csv(directory / f"{prefix}{SUMMARY_FILENAME}", index=False)
    return summary, directory
