"""Element-level reference tables, shared by every dataset.

These are not datasets: they carry per-element physical properties that the descriptors in prepdata/ are computed
against, so any dataset added under loaddata/ reads its own file but reuses these unchanged.

Adapted from https://github.com/rich970/ML-alloy-design/blob/master/alloys.py with modifications.
"""

from typing import Tuple

import pandas as pd

DEFAULT_PERIODIC_TABLE_PATH = "./data/Periodic-table/periodic_table.xlsx"
DEFAULT_MIEDEMA_PATH = "./data/Miedema-model/Miedema-model-reduced.xlsx"


def load_periodic_table(path: str = DEFAULT_PERIODIC_TABLE_PATH) -> pd.DataFrame:
    """Read the periodic table spreadsheet, indexed by element symbol."""
    pt = pd.read_excel(path)
    pt.index = pt["symbol"]
    return pt


def load_miedema_weight(path: str = DEFAULT_MIEDEMA_PATH) -> pd.DataFrame:
    """Read the Miedema mixing-enthalpy spreadsheet and symmetrize it.

    The sheet stores only one triangle of the element-pair matrix, so it is added to its own transpose to make
    `mm.loc[a, b]` and `mm.loc[b, a]` both resolve.
    """
    mm = pd.read_excel(path, header=1, index_col=73, usecols=range(0, 74), nrows=73).fillna(0)
    return mm + mm.transpose().fillna(0)


def load_element_properties(
    pt_path: str = DEFAULT_PERIODIC_TABLE_PATH,
    mm_path: str = DEFAULT_MIEDEMA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both reference tables at once, since the descriptors always need the pair.

    Returns:
        (periodic_table, miedema_weight).
    """
    return load_periodic_table(pt_path), load_miedema_weight(mm_path)
