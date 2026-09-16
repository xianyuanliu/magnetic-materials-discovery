"""Assembly of the modeling table: run the descriptors, then decide which measured rows survive.

Featurization is separate (prepdata/alloy_descriptors.py) because inference has compositions but no target:
pipeline/inference_pipeline.py and interpret/case_studies.py need the same nine features for formulas that have never
been measured, without any of the target-side filtering below.

The filtering here is task selection, not dataset cleaning — it drops rows this task cannot use. Rows a dataset should
never have offered are dropped by its loaddata/ module instead.
"""

from typing import List, Tuple

import pandas as pd

from prepdata.alloy_descriptors import ENGINEERED_FEATURE_COLUMNS, add_engineered_features


def build_modeling_table(
    raw_data: pd.DataFrame,
    pt: pd.DataFrame,
    mm: pd.DataFrame,
    min_target: float = 0.18,
    target_column: str = "saturation magnetization",
    formula_column: str = "chemical formula",
) -> Tuple[pd.DataFrame, List[str]]:
    """Run the descriptors over `raw_data`, then keep the rows this task can model.

    Args:
        raw_data: Frame carrying `formula_column` and `target_column`.
        pt: Periodic table data, indexed by element symbol.
        mm: Symmetrized Miedema mixing-enthalpy matrix.
        min_target: Rows whose saturation magnetization is below this are dropped as non-magnetic. The default cutoff
            was determined through prior model optimization.
        target_column: Name of the measured property being modeled.
        formula_column: Name of the chemical-formula column.

    Returns:
        (data, feature_columns), where `data` is indexed by formula with one row per distinct composition
        (duplicates collapsed by median).
    """
    data = add_engineered_features(raw_data, pt, mm, formula_column=formula_column)

    feature_columns = list(ENGINEERED_FEATURE_COLUMNS)
    data = data.dropna(axis=0, subset=[target_column] + feature_columns)
    data = data[data[target_column] >= min_target]

    # Collapse duplicate formulas by taking the median feature values.
    data = data.groupby(by=formula_column).median()

    return data, feature_columns
