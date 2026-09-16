"""Pins the reusable composition primitives and the nine descriptor values.

These are exact-value checks, not tolerances: a refactor of prepdata/ is expected to leave every number
bit-identical, so any drift should fail here rather than be discovered by comparing saved CSVs later.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from loaddata.tabular_access import load_element_properties
from prepdata.alloy_descriptors import ENGINEERED_FEATURE_COLUMNS, add_engineered_features
from prepdata.composition import (
    formula_contains_elements,
    get_atomic_fraction,
    get_atomic_fraction_array,
    get_compound_radix,
    get_elements,
    get_elements_per_row,
    get_group_period_maps,
    get_mixing_entropy,
    get_stoich_array,
    get_weighted_property,
)

# Nine descriptors per formula, in ENGINEERED_FEATURE_COLUMNS order, from the committed reference tables.
EXPECTED_FEATURES = {
    "Fe": (1.0, -0.0, 55.8452, 4.0, 8.0, 1811.0, 0.0, 6.0, 1.83),
    "FeNi": (2.0, 0.6931471805599453, 57.26932, 4.0, 9.0, 1769.5, -2.0, 5.0, 1.87),
    "Nd2Fe14B": (3.0, 0.5783252866601273, 63.5957294117647, 4.117647058823529, 7.705882352941176,
                 1782.1764705882351, -6.006920415224914, 5.470588235294118, 1.7611764705882353),
    "AlCo2Cr": (3.0, 1.0397207708399179, 49.2110218425, 3.75, 9.25, 1662.3675, -14.0, 4.75, 1.7575),
}


def small_periodic_table():
    """Two-element reference tables, enough to exercise the primitives without reading a spreadsheet."""
    pt = pd.DataFrame({
        "symbol": ["Fe", "Ni"], "atomic_weight": [55.845, 58.6934], "period": [4, 4],
        "group_block": ["group 8, d-block", "group 10, d-block"], "melting_point": [1811, 1728],
        "valence": [8, 10], "electronegativity": [1.83, 1.91],
    }).set_index("symbol", drop=False)
    return pt


class DescriptorValueTests(unittest.TestCase):
    """Exact descriptor values against the committed periodic table and Miedema sheet."""

    @classmethod
    def setUpClass(cls):
        cls.pt, cls.mm = load_element_properties()

    def test_nine_descriptors_match_pinned_values(self):
        frame = pd.DataFrame({"chemical formula": list(EXPECTED_FEATURES)})
        out = add_engineered_features(frame, self.pt, self.mm).set_index("chemical formula")
        for formula, expected in EXPECTED_FEATURES.items():
            got = tuple(float(out.loc[formula, column]) for column in ENGINEERED_FEATURE_COLUMNS)
            self.assertEqual(got, expected, f"{formula} descriptors drifted")

    def test_scaled_formula_gives_identical_descriptors(self):
        """Descriptors depend on atomic fractions only, so Fe0.5Ni0.5 must equal FeNi exactly."""
        frame = pd.DataFrame({"chemical formula": ["FeNi", "Fe0.5Ni0.5", "Fe2Ni2"]})
        out = add_engineered_features(frame, self.pt, self.mm)[list(ENGINEERED_FEATURE_COLUMNS)]
        first = out.iloc[0].to_numpy(float)
        for row in range(1, len(out)):
            self.assertTrue(np.array_equal(out.iloc[row].to_numpy(float), first))

    def test_unparseable_formula_yields_nan_not_zero(self):
        """A fabricated all-zero feature vector would train the model on a lie; NaN gets the row dropped."""
        frame = pd.DataFrame({"chemical formula": ["FeNi", "NotAFormula!!"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = add_engineered_features(frame, self.pt, self.mm)
        bad = out.iloc[1][list(ENGINEERED_FEATURE_COLUMNS)]
        self.assertTrue(bad.isna().all(), f"expected all NaN, got {bad.to_dict()}")

    def test_group_period_maps_omit_elements_without_a_group(self):
        """The f-block has no group number in the sheet; it must be absent, not present as NaN."""
        element_to_group, element_to_period = get_group_period_maps(self.pt)
        self.assertEqual(element_to_group["Fe"], 8)
        self.assertEqual(element_to_period["Fe"], 4)
        self.assertNotIn("U", element_to_group)
        self.assertIn("U", element_to_period)
        self.assertTrue(all(isinstance(v, int) for v in element_to_group.values()))


class CompositionPrimitiveTests(unittest.TestCase):
    """The reusable pieces another materials task builds its own descriptors from."""

    def setUp(self):
        self.pt = small_periodic_table()

    def test_stoich_array_keeps_fractional_amounts(self):
        """Casting to int would floor Fe0.5Ni0.5 to an all-zero row that silently features as 0.0."""
        frame = pd.DataFrame({"chemical formula": ["FeNi", "Fe0.5Ni0.5", "Fe3Ni"]})
        stoich = get_stoich_array(frame, self.pt)
        self.assertEqual(list(stoich.loc[0, ["Fe", "Ni"]]), [1.0, 1.0])
        self.assertEqual(list(stoich.loc[1, ["Fe", "Ni"]]), [0.5, 0.5])
        self.assertEqual(list(stoich.loc[2, ["Fe", "Ni"]]), [3.0, 1.0])

    def test_atomic_fraction_normalizes_and_reports_empty(self):
        stoich = get_stoich_array(pd.DataFrame({"chemical formula": ["Fe3Ni"]}), self.pt)
        fraction = get_atomic_fraction(stoich.iloc[0])
        self.assertEqual(fraction["Fe"], 0.75)
        self.assertEqual(fraction["Ni"], 0.25)
        self.assertTrue(get_atomic_fraction(stoich.iloc[0] * 0).empty)

    def test_atomic_fraction_array_keeps_the_input_index(self):
        """An unusable row yields an unnamed Series, which pandas would relabel and silently misalign."""
        frame = pd.DataFrame({"chemical formula": ["FeNi", "NotAFormula!!", "Fe3Ni"]}, index=["a", "b", "c"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stoich = get_stoich_array(frame, self.pt)
            fractions = get_atomic_fraction_array(stoich)
        self.assertEqual(list(fractions.index), ["a", "b", "c"])
        self.assertTrue(fractions.loc["b"].isna().all())
        self.assertEqual(fractions.loc["c", "Fe"], 0.75)

    def test_weighted_property_is_the_fraction_weighted_mean(self):
        stoich = get_stoich_array(pd.DataFrame({"chemical formula": ["Fe3Ni"]}), self.pt)
        weighted = get_weighted_property(self.pt["atomic_weight"], stoich)
        self.assertEqual(weighted.iloc[0], 0.75 * 55.845 + 0.25 * 58.6934)

    def test_mixing_entropy_matches_the_closed_form(self):
        stoich = get_stoich_array(pd.DataFrame({"chemical formula": ["FeNi", "Fe3Ni"]}), self.pt)
        entropy = get_mixing_entropy(stoich)
        self.assertEqual(entropy.iloc[0], -np.dot([0.5, 0.5], np.log([0.5, 0.5])))
        self.assertEqual(entropy.iloc[1], -np.dot([0.75, 0.25], np.log([0.75, 0.25])))

    def test_compound_radix_counts_distinct_elements(self):
        frame = pd.DataFrame({"chemical formula": ["Fe", "FeNi", "Fe3Ni", "NotAFormula!!", None]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            radix = get_compound_radix(frame)
        self.assertEqual(list(radix[:3]), [1.0, 2.0, 2.0])
        self.assertTrue(radix[3:].isna().all())

    def test_element_lookup_ignores_substring_matches(self):
        """String matching would find S inside Si; parsing must not."""
        frame = pd.DataFrame({"chemical formula": ["FeSi", "FeS", "CoNi"]})
        self.assertEqual(get_elements("FeSi"), ["Fe", "Si"])
        self.assertEqual(list(formula_contains_elements(frame, ["S"])), [False, True, False])
        self.assertEqual(list(formula_contains_elements(frame, ["Fe", "Co"])), [True, True, True])

    def test_elements_per_row_is_row_aligned_and_quiet_on_missing(self):
        frame = pd.DataFrame({"chemical formula": ["FeNi", None, "Fe3Ni"]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = get_elements_per_row(frame)
        self.assertEqual(rows, [["Fe", "Ni"], [], ["Fe", "Ni"]])
        self.assertEqual([w for w in caught if "could not be parsed" in str(w.message)], [])


if __name__ == "__main__":
    unittest.main()
