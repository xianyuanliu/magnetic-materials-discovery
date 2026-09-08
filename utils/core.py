"""Shared types and defaults with no dependency on any stage package.

This module is deliberately a leaf: it imports nothing from `loaddata`,
`prepdata`, `pipeline`, `evaluate` or `interpret`. Putting the vocabulary they
all share here is what lets `evaluate/` score results without importing
`pipeline/`, which previously made the two packages mutually dependent.
"""

from typing import Tuple

import numpy as np

# A train/test split: an identifier plus the row positions on each side.
# Split builders live in pipeline/ood_splits.py; the evaluators only consume
# splits, so they annotate against this alias rather than importing that module.
Split = Tuple[str, np.ndarray, np.ndarray]

# Regression metrics carried through every result table, and the decimals each
# is printed with. MRE is a small fraction, so it needs more places.
METRICS = ("mse", "mae", "mre", "r2")
METRIC_DECIMALS = {"mse": 4, "mae": 4, "mre": 6, "r2": 4}

# Nominal miscoverage of a reported prediction interval; 0.05 gives the usual 95%.
DEFAULT_ALPHA = 0.05
