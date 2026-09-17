"""Where a run's numbers go: one raw table and one summary, written the same way by every evaluation mode.

The raw table is the run's primary artifact — every figure, significance test and derived table in the repo is
computed from it, so a run that saves it can be re-analysed without being repeated. Saving is opt-out rather than
opt-in for that reason.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from evaluate.metrics import SUMMARY_GROUPS, summarize_scores

RESULTS_FILENAME = "results.csv"
SUMMARY_FILENAME = "summary.csv"


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
