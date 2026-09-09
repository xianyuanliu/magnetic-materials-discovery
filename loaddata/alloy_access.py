"""Raw file readers for Novamag JSON records, the periodic table, and Miedema data.

Adapted from https://github.com/rich970/ML-alloy-design/blob/master/alloys.py with modifications.
"""

import os

import pandas as pd


def _flatten(x):
    """Flatten nested dicts that only contain a single 'value' field."""
    if isinstance(x, dict) and "value" in x and len(x) == 1:
        return x["value"]
    return x


def import_novamag(root_dir):
    """Load all Novamag JSON files into a flat DataFrame of chemistry, crystal, and magnetic properties."""

    rows = []
    failed_files = []

    for dirName, subdirList, fileList in os.walk(root_dir):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname.endswith(".json"):
                filepath = os.path.join(dirName, fname)
                try:
                    df = pd.read_json(filepath, encoding="Latin")
                    row = {**df.properties.chemistry, **df.properties.crystal, **df.properties.magnetics}
                    rows.append(row)
                except ValueError:
                    print("Import failed", filepath)
                    failed_files.append(filepath)
    X = pd.DataFrame(rows)
    X = X.apply(lambda col: col.map(_flatten))
    return X


def import_periodic_table(root_dir):
    """Import the periodic table spreadsheet and index by symbol."""
    pt = pd.read_excel(root_dir)
    pt.index = pt["symbol"]
    return pt


def import_miedema_weight(root_dir):
    """Import Miedema model enthalpies spreadsheet and symmetrise it."""
    mm = pd.read_excel(root_dir, header=1, index_col=73, usecols=range(0, 74), nrows=73).fillna(0)
    mm_T = mm.transpose().fillna(0)
    mm = mm + mm_T
    return mm
