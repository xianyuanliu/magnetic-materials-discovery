"""Covers the branches that fail quietly: renormalized descriptors, missing reference data, and feature selection.

Guard clauses that raise on a missing column are deliberately left out — they announce themselves. What is pinned here
is behavior that would otherwise produce a plausible-looking number, a silently shorter table, or a model input that
should never have been one.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from loaddata.tabular_access import resolve_feature_columns
from prepdata.alloy_descriptors import get_electronegw, get_groupw, get_melting_tw, get_miedemaw
from prepdata.composition import get_composition_key, get_stoich_array
from prepdata.feature_table import build_feature_table


def periodic_table_with_f_block():
    """Fe and Ni carry group numbers; Nd is spelled the way the sheet spells the f-block, with no group."""
    return pd.DataFrame({
        "symbol": ["Fe", "Ni", "Nd"], "atomic_weight": [55.845, 58.6934, 144.242], "period": [4, 4, 6],
        "group_block": ["group 8, d-block", "group 10, d-block", "group n/a, f-block"],
        "melting_point": [1811, 1728, 1297], "valence": [8, 10, 3],
        "electronegativity": ["1.83", "1.91", "1.14"],
    }).set_index("symbol", drop=False)


class RenormalizedDescriptorTests(unittest.TestCase):
    """get_groupw is the one descriptor that drops elements and renormalizes, so its weights need pinning."""

    def setUp(self):
        self.pt = periodic_table_with_f_block()

    def test_f_block_is_dropped_and_the_rest_renormalized(self):
        """Nd has no group number: Fe3Nd1 must weigh Fe alone, not dilute it toward zero."""
        stoich = get_stoich_array(pd.DataFrame({"chemical formula": ["Fe3Nd", "Fe3Ni"]}), self.pt)
        groupw = get_groupw(self.pt, stoich)
        self.assertEqual(groupw.iloc[0], 8.0)
        self.assertEqual(groupw.iloc[1], 0.75 * 8 + 0.25 * 10)

    def test_composition_of_only_ungrouped_elements_is_nan(self):
        stoich = get_stoich_array(pd.DataFrame({"chemical formula": ["Nd"]}), self.pt)
        self.assertTrue(np.isnan(get_groupw(self.pt, stoich).iloc[0]))


class MissingReferenceDataTests(unittest.TestCase):
    """A pair absent from the Miedema sheet must give NaN; a 0.0 would read as 'no mixing enthalpy'."""

    def test_element_pair_missing_from_miedema_yields_nan(self):
        pt = periodic_table_with_f_block()
        mm = pd.DataFrame([[0.0, -2.0], [-2.0, 0.0]], index=["Fe", "Ni"], columns=["Fe", "Ni"])
        stoich = get_stoich_array(pd.DataFrame({"chemical formula": ["FeNi", "FeNd"]}), pt)
        miedemaw = get_miedemaw(mm, stoich)
        self.assertEqual(miedemaw.iloc[0], 4 * 0.5 * 0.5 * -2.0)
        self.assertTrue(np.isnan(miedemaw.iloc[1]), "a missing pair must not fall back to 0.0")

    def test_build_feature_table_reports_what_it_dropped(self):
        """Losing rows to incomplete reference data is a data-quality signal, not an implementation detail."""
        pt = periodic_table_with_f_block()
        mm = pd.DataFrame([[0.0, -2.0], [-2.0, 0.0]], index=["Fe", "Ni"], columns=["Fe", "Ni"])
        records = pd.DataFrame({
            "chemical formula": ["FeNi", "FeNd"],
            "saturation magnetization": [1.5, 1.5],
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table, _ = build_feature_table(records, pt, mm)
        self.assertEqual(list(table.index), ["FeNi"])
        messages = [str(w.message) for w in caught if "Dropped" in str(w.message)]
        self.assertEqual(len(messages), 1, f"expected one drop warning, got {messages}")
        self.assertIn("miedemaH", messages[0])


class AnnotatedPropertyColumnTests(unittest.TestCase):
    """The sheet stores two properties as prose, so the descriptors must read the number out of it."""

    def test_annotated_melting_point_is_parsed_not_crashed_on(self):
        """Np is spelled "912±3 K (639±3 °C, 1182±5 °F)"; passing that to np.dot raises TypeError."""
        pt = periodic_table_with_f_block()
        pt.loc["Nd", "melting_point"] = "912\u00b13 K (639\u00b13 \u00b0C, 1182\u00b15 \u00b0F)"
        stoich = get_stoich_array(pd.DataFrame({"chemical formula": ["NdFe"]}), pt)
        self.assertEqual(get_melting_tw(pt, stoich).iloc[0], 0.5 * 912 + 0.5 * 1811)

    def test_property_without_a_number_becomes_nan(self):
        """"Pauling scale: no data" must not silently weigh as zero."""
        pt = periodic_table_with_f_block()
        pt.loc["Nd", "electronegativity"] = "Pauling scale: no data"
        stoich = get_stoich_array(pd.DataFrame({"chemical formula": ["NdFe"]}), pt)
        self.assertTrue(np.isnan(get_electronegw(pt, stoich).iloc[0]))

    def test_already_numeric_column_is_accepted(self):
        """A caller supplying a plain numeric periodic table should not need the sheet's prose format."""
        pt = periodic_table_with_f_block()
        pt["electronegativity"] = [1.83, 1.91, 1.14]
        stoich = get_stoich_array(pd.DataFrame({"chemical formula": ["FeNi"]}), pt)
        self.assertEqual(get_electronegw(pt, stoich).iloc[0], 0.5 * 1.83 + 0.5 * 1.91)

    def test_numeric_column_is_not_routed_through_text_extraction(self):
        """str(1e-07) is "1e-07", whose leading number is 1.0 — seven orders of magnitude wrong."""
        pt = periodic_table_with_f_block()
        pt["melting_point"] = [1e-7, 3e-7, 1.0]
        stoich = get_stoich_array(pd.DataFrame({"chemical formula": ["FeNi"]}), pt)
        self.assertEqual(get_melting_tw(pt, stoich).iloc[0], 0.5 * 1e-7 + 0.5 * 3e-7)


class FeatureColumnSelectionTests(unittest.TestCase):
    """resolve_feature_columns is what stops a stray id or text column from becoming a model input."""

    def setUp(self):
        # row_index and note stand for columns this project has never seen, which the fallback cannot recognize.
        self.data = pd.DataFrame({
            "chemical formula": ["FeNi"], "saturation magnetization": [1.5],
            "Zw": [57.3], "row_index": [7], "note": ["measured twice"],
        })

    def test_named_columns_are_used_in_the_given_order(self):
        self.assertEqual(
            resolve_feature_columns(self.data, "saturation magnetization", feature_columns=["Zw"]), ["Zw"]
        )

    def test_fallback_rejects_a_non_numeric_column_instead_of_passing_it_through(self):
        with self.assertRaises(ValueError) as caught:
            resolve_feature_columns(self.data, "saturation magnetization")
        self.assertIn("note", str(caught.exception))

    def test_fallback_promotes_an_unrecognized_numeric_column(self):
        """row_index becomes a model input, which is why naming the features explicitly is the documented default."""
        numeric = self.data.drop(columns=["note"])
        self.assertEqual(
            resolve_feature_columns(numeric, "saturation magnetization"), ["Zw", "row_index"]
        )

    def test_fallback_skips_the_provenance_columns(self):
        """sample_id and n_records are carried for traceability; promoting them would leak row provenance."""
        data = self.data.drop(columns=["note"]).assign(sample_id=["a/b.json"], n_records=[3], source=["novamag"])
        self.assertEqual(
            resolve_feature_columns(data, "saturation magnetization"), ["Zw", "row_index"]
        )

    def test_unknown_named_column_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_feature_columns(self.data, "saturation magnetization", feature_columns=["nope"])


class SingleFormulaAndOverflowTests(unittest.TestCase):
    """Documented input shapes and the amount guards, neither of which the dataset paths reach."""

    def test_stoich_array_accepts_a_bare_formula_string(self):
        pt = periodic_table_with_f_block()
        stoich = get_stoich_array("Fe3Ni", pt)
        self.assertEqual(len(stoich), 1)
        self.assertEqual(list(stoich.iloc[0][["Fe", "Ni"]]), [3.0, 1.0])

    def test_non_finite_amounts_are_rejected(self):
        """pymatgen accepts Fe1e400 as an infinite amount, and two finite amounts can still sum to infinity."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsNone(get_composition_key("Fe1e400"))
            self.assertIsNone(get_composition_key("Fe1e308Ni1e308"))
            self.assertIsNone(get_composition_key("Fe0"))


if __name__ == "__main__":
    unittest.main()
