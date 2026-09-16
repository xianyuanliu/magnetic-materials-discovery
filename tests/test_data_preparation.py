"""Regression checks for dataset boundaries, composition identity and magnetic-subset targets."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from loaddata.data_access import load_features_and_target, standardize_records
from loaddata.materials_project import TESLA_PER_BOHR_MAGNETON_PER_CUBIC_ANGSTROM, load_materials_project
from loaddata.novamag import load_novamag
from prepdata.alloy_descriptors import ENGINEERED_FEATURE_COLUMNS
from prepdata.composition import get_elements, get_normalized_formula
from prepdata.modeling_table import build_modeling_table, filter_samples


def reference_tables():
    pt = pd.DataFrame({
        "symbol": ["Fe", "Ni"], "atomic_weight": [55.845, 58.6934], "period": [4, 4],
        "group_block": ["group 8, d-block", "group 10, d-block"], "melting_point": [1811, 1728],
        "valence": [8, 10], "electronegativity": ["1.83", "1.91"],
    }).set_index("symbol", drop=False)
    mm = pd.DataFrame([[0.0, -2.0], [-2.0, 0.0]], index=["Fe", "Ni"], columns=["Fe", "Ni"])
    return pt, mm


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tables = reference_tables()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_equivalent_formulas_share_key_without_merging_different_ratios(self):
        formulas = ["FeNi", "Fe2Ni2", "NiFe", "Fe0.5Ni0.5", "Ni0.1Fe0.1", "(FeNi)2"]
        assert {get_normalized_formula(formula) for formula in formulas} == {"FeNi"}
        assert get_normalized_formula("Fe2Ni") != get_normalized_formula("FeNi")
        assert get_normalized_formula("Fe2Ni") == get_normalized_formula("Ni2Fe4")
        assert get_normalized_formula("Fe") == get_normalized_formula("Fe8")
        for formula in [*formulas, "CoFe4Ta", "CoFe4Ta2", "Fe0.1Ni0.9", "(Fe0.1Ni0.2)3"]:
            key = get_normalized_formula(formula)
            assert get_normalized_formula(key) == key

    def test_invalid_compositions_do_not_form_a_group(self):
        assert get_normalized_formula(None) is None
        assert get_normalized_formula(np.nan) is None
        assert get_normalized_formula(pd.NA) is None
        with self.assertWarnsRegex(UserWarning, "Could not normalize"):
            assert get_normalized_formula("invalid") is None

    def test_sodium_nitrogen_key_survives_default_csv_reading(self):
        key = get_normalized_formula("NaN")
        assert key == "NNa"
        assert get_normalized_formula("Na2N2") == key
        assert get_normalized_formula(key) == key
        path = self.tmp_path / "formula.csv"
        pd.DataFrame({"chemical formula": [key]}).to_csv(path, index=False)
        loaded_formula = pd.read_csv(path)["chemical formula"].iloc[0]
        assert loaded_formula == key
        assert set(get_elements(loaded_formula)) == {"Na", "N"}

    def test_filter_before_median_and_aggregate_only_target(self):
        reference_tables = self.tables
        tmp_path = self.tmp_path
        raw = pd.DataFrame({
            "chemical formula": ["FeNi", "Fe2Ni2", "NiFe", "Fe0.5Ni0.5", "Fe2Ni"],
            "saturation magnetization": [0.05, 0.10, 1.5, 2.5, 0.8],
            "sample_id": ["a", "b", "c", "d", "e"], "space_group": [1, 2, 3, 4, 5],
        })
        original = raw.copy(deep=True)
        table, features = build_modeling_table(raw, *reference_tables)
        assert len(table) == 2
        # Aggregating first would give 0.8 for FeNi, whereas the selected magnetic subset gives 2.0.
        assert table.loc["FeNi", "saturation magnetization"] == 2.0
        self.assertAlmostEqual(table.loc["FeNi", "Zw"], (55.845 + 58.6934) / 2)
        assert table.loc["FeNi", "miedemaH"] == -2.0
        assert list(table.columns) == ["saturation magnetization", *ENGINEERED_FEATURE_COLUMNS]
        assert_frame_equal(raw, original)
        path = tmp_path / "prepared.csv"
        table.to_csv(path)
        X, y, resolved = load_features_and_target(path, feature_columns=features)
        assert resolved == features
        assert len(X) == len(y) == 2
        assert np.isfinite(X.to_numpy()).all()

    def test_selection_parameters_and_missing_labels(self):
        raw = standardize_records(pd.DataFrame({
            "chemical formula": ["FeNi", "Fe", "NdFe", "UFe", "Ni", "Fe2Ni"],
            "saturation magnetization": [0.18, 1, 1, 1, 1, 0.1],
            "is_magnetic": [True, False, True, True, None, True],
        }))
        selected = filter_samples(raw, excluded_elements=["Nd", "U"], magnetic_only=True)
        assert selected["chemical formula"].tolist() == ["FeNi"]
        assert len(filter_samples(raw, min_target=None)) == 6
        with self.assertRaisesRegex(ValueError, "is_magnetic"):
            filter_samples(raw.drop(columns="is_magnetic"), magnetic_only=True)

    def test_standardization_preserves_metadata_and_rejects_unknown_boolean(self):
        raw = pd.DataFrame({
            "chemical formula": [" FeNi ", "Fe"], "saturation magnetization": ["1.2", "inf"],
            "is_magnetic": ["TRUE", "false"], "sample_id": ["a", "b"],
        })
        data = standardize_records(raw)
        assert data["sample_id"].tolist() == ["a", "b"]
        assert data["chemical formula"].iloc[0] == "FeNi"
        assert data["saturation magnetization"].iloc[0] == 1.2
        assert pd.isna(data["saturation magnetization"].iloc[1])
        assert data["is_magnetic"].tolist() == [True, False]
        raw.loc[0, "is_magnetic"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown is_magnetic"):
            standardize_records(raw)

    def test_mp_loader_keeps_nonmagnetic_and_excluded_element_records(self):
        tmp_path = self.tmp_path
        path = tmp_path / "mp.csv"
        pd.DataFrame({
            "composition": ["FeNi", "NdFe", "UFe"], "material_id": ["mp-a", "mp-b", "mp-c"],
            "total_magnetization_normalized_vol": [0.1, 0.2, 0.3], "is_magnetic": [False, True, True],
        }).to_csv(path, index=False)
        data = load_materials_project(path, formula_column="formula", target_column="target")
        assert data["formula"].tolist() == ["FeNi", "NdFe", "UFe"]
        assert data["sample_id"].tolist() == ["mp-a", "mp-b", "mp-c"]
        self.assertAlmostEqual(data["target"].iloc[0], 0.1 * TESLA_PER_BOHR_MAGNETON_PER_CUBIC_ANGSTROM)
        assert data["is_magnetic"].tolist() == [False, True, True]

    def test_novamag_loader_standardizes_source_fields_and_preserves_metadata(self):
        tmp_path = self.tmp_path
        (tmp_path / "record.json").write_text(json.dumps({"properties": {
            "chemistry": {"chemical formula": {"value": "Fe2Ni2"}},
            "crystal": {"compound space group": {"value": 1}},
            "magnetics": {"saturation magnetization": {"value": "0.10"}},
        }}))
        data = load_novamag(tmp_path, formula_column="formula", target_column="target")
        assert data["formula"].tolist() == ["Fe2Ni2"]
        assert data["target"].tolist() == [0.1]  # Loader does not impose the magnetic-subset threshold.
        assert data["compound space group"].tolist() == [1]
        assert data["sample_id"].tolist() == ["record.json"]
        with self.assertRaises(FileNotFoundError):
            load_novamag(tmp_path / "missing")

    def test_empty_selection_has_valid_output_schema(self):
        reference_tables = self.tables
        raw = pd.DataFrame({"chemical formula": ["FeNi"], "saturation magnetization": [0.05]})
        table, features = build_modeling_table(raw, *reference_tables)
        assert table.empty
        assert table.index.name == "chemical formula"
        assert list(table.columns) == ["saturation magnetization", *features]


if __name__ == "__main__":
    unittest.main()
