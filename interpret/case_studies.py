"""Sweeping a model along an alloy series, so its predictions can be read against measured values.

Given formulas spanning a binary series, this featurizes them with the same code the models were trained with,
predicts along the series, and hands back panels for interpret/reference_comparison.py to draw. Sharing
`add_engineered_features` with training is the point: a case study that featurized differently would be comparing the
model against a version of itself it has never seen.

No series is defined here. Which alloys are worth sweeping, and what has been measured for them, belongs to whatever
study is asking — this module only knows how to turn such a series into predictions.
"""

from typing import Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple

import pandas as pd

from interpret.reference_comparison import ComparisonPanel, panel_from_series
from prepdata.alloy_descriptors import add_engineered_features
from prepdata.composition import get_atomic_fraction_array, get_stoich_array
from utils.persistence import align_features


class AlloySeries(NamedTuple):
    """One binary series to sweep a model along, and what has been measured for it.

    Attributes:
        title: Display name, e.g. "FeCo".
        element: Alloying element whose atomic fraction is the x axis.
        formulas: Chemical formulas spanning the series.
        measurements: Measured target values, keyed by the alloying element's atomic fraction. These come from
            experiment, so their fractions are whatever was actually made and need not match `formulas`.
    """

    title: str
    element: str
    formulas: List[str]
    measurements: Dict[float, float]


def build_series_features(
    formulas: Sequence[str],
    periodic_table: pd.DataFrame,
    miedema_weight: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature matrix for a list of chemical formulas.

    Shares prepdata.alloy_descriptors.add_engineered_features with training, so these features cannot drift from the
    ones the models were fitted on.

    Args:
        formulas: Chemical formulas to featurize.
        periodic_table: Periodic table data, indexed by element symbol.
        miedema_weight: Symmetrized Miedema mixing-enthalpy matrix.
        formula_column: Name to give the formula column.

    Returns:
        (features, stoich_array). `features` carries the formula column plus the engineered features; `stoich_array`
        is the per-element stoichiometry they were derived from.
    """
    frame = pd.DataFrame({formula_column: list(formulas)})
    stoich = get_stoich_array(frame, periodic_table, formula_column=formula_column)
    features = add_engineered_features(frame, periodic_table, miedema_weight, formula_column=formula_column)
    return features, stoich


def build_series_panel(
    series: AlloySeries,
    feature_columns: Sequence[str],
    models: Mapping[str, Any],
    periodic_table: pd.DataFrame,
    miedema_weight: pd.DataFrame,
) -> ComparisonPanel:
    """Sweep every model along `series` and pair the result with its measurements.

    Args:
        series: The alloy series to evaluate.
        feature_columns: Feature columns the models were trained on, in order.
        models: Fitted models keyed by display name.
        periodic_table: Periodic table data, indexed by element symbol.
        miedema_weight: Symmetrized Miedema mixing-enthalpy matrix.

    Returns:
        A panel ready for interpret.reference_comparison.plot_predictions_against_measurements.
    """
    features, stoich_array = build_series_features(series.formulas, periodic_table, miedema_weight)
    X = align_features(features, feature_columns, source=f"the {series.title} case study")
    atomic_fraction = get_atomic_fraction_array(stoich_array)

    return panel_from_series(
        title=f"{series.title} Case Study",
        x_label=f"{series.element} content [atomic fraction]",
        x=atomic_fraction[series.element],
        predictions={name: model.predict(X) for name, model in models.items()},
        measurements=series.measurements,
    )


def build_series_panels(
    all_series: Sequence[AlloySeries],
    feature_columns: Sequence[str],
    models: Mapping[str, Any],
    periodic_table: pd.DataFrame,
    miedema_weight: pd.DataFrame,
) -> List[ComparisonPanel]:
    """Build one panel per series, in the order given.

    Raises:
        ValueError: If no model is given, since every panel would then be empty.
    """
    if not models:
        raise ValueError("A case study needs at least one fitted model.")
    return [
        build_series_panel(series, feature_columns, models, periodic_table, miedema_weight)
        for series in all_series
    ]
