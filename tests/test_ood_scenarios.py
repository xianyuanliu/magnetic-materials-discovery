"""Pins which scenarios `build_scenarios` builds for a given selection.

A selection that silently builds nothing produces an empty run that still writes its (empty) tables and exits 0.
"""

import unittest

import numpy as np
import pandas as pd

from pipeline.ood import build_scenarios

ALL = ("element", "period", "group", "cluster", "sparsex", "sparsey")


def toy_dataset(n_rows: int = 60):
    """Rows alternating between Fe-Co and Fe-Ni; Co is given its own period so every family has two sides."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n_rows, 3)), columns=list("abc"))
    y = pd.Series(np.arange(n_rows, dtype=float))
    elements = [["Fe", "Co"] if i % 2 else ["Fe", "Ni"] for i in range(n_rows)]
    group = {"Fe": 8, "Co": 9, "Ni": 10}
    period = {"Fe": 4, "Co": 5, "Ni": 4}
    return X, y, elements, group, period


class BuildScenariosTests(unittest.TestCase):
    def test_every_scenario_builds_by_default(self):
        names = [name for name, _ in build_scenarios(*toy_dataset(), min_train=5, min_test=5)]
        self.assertEqual(names, ["LOEO", "LOPO", "LOGO", "LOCO(k=5)", "SparseX", "SparseY"])

    def test_a_subset_builds_only_those(self):
        built = build_scenarios(*toy_dataset(), scenarios=("sparsey", "element"), min_train=5, min_test=5)
        names = [name for name, _ in built]
        self.assertEqual(names, ["LOEO", "SparseY"])

    def test_an_empty_selection_builds_nothing(self):
        self.assertEqual(build_scenarios(*toy_dataset(), scenarios=(), min_train=5, min_test=5), [])


if __name__ == "__main__":
    unittest.main()
