"""Analyze saved results, and save model-explanation and case-study figures.

`analyze_results` reads only the result, summary and split-size tables, so it applies to any evaluation
workflow and to results saved by an earlier run.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import pandas as pd

from kalematerials.embed.composition_descriptors import add_engineered_features
from kalematerials.interpret.model_explanations import plot_model_explanations
from kalematerials.interpret.reference_comparison import (
    ComparisonPanel,
    panel_from_series,
    plot_predictions_against_measurements,
)
from kalematerials.interpret.result_analysis import summarize_generalization_gap, summarize_model_comparison
from kalematerials.interpret.result_figures import plot_generalization_gap, plot_metric_by_model
from kalematerials.loaddata.element_tables import load_element_properties
from kalematerials.prepdata.composition import get_atomic_fraction_array, get_stoich_array
from kalematerials.utils.persistence import align_features


class ResultAnalysis(NamedTuple):
    """Tables derived from saved results; each is empty when it does not apply.

    `significance` compares the two `compare` models (interpret.result_analysis.summarize_model_comparison).
    `generalization_gap` needs OOD and ID-reference rows (interpret.result_analysis.summarize_generalization_gap).
    """

    significance: pd.DataFrame
    generalization_gap: pd.DataFrame


def analyze_results(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    compare: Optional[Tuple[str, str]] = None,
) -> ResultAnalysis:
    """Derive the significance and generalization-gap tables that the results support.

    `compare` names two models by display name; without it the significance table is empty.
    """
    return ResultAnalysis(
        significance=(
            summarize_model_comparison(results, splits, *compare) if compare is not None else pd.DataFrame()
        ),
        generalization_gap=summarize_generalization_gap(summary),
    )


def plot_results(
    results: pd.DataFrame,
    analysis: ResultAnalysis,
    *,
    plots_dir: Path,
    prefix: str = "",
    metric: str = "mse",
) -> List[Path]:
    """Save the result figures the tables support and return their paths.

    Always draws `<prefix><metric>_by_model.png`; draws `<prefix>generalization_gap.png` when the analysis
    has a generalization-gap table.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved = [plots_dir / f"{prefix}{metric}_by_model.png"]
    plot_metric_by_model(results, metric=metric, save_path=saved[0])
    if not analysis.generalization_gap.empty:
        saved.append(plots_dir / f"{prefix}generalization_gap.png")
        plot_generalization_gap(analysis.generalization_gap, save_path=saved[-1])
    return saved


class AlloySeries(NamedTuple):
    """Binary alloy formulas and reference measurements.

    `element` is the alloying element on the x-axis. `measurements` maps its atomic fraction to measured
    targets; measurement compositions may differ from `formulas`.
    """

    title: str
    element: str
    formulas: List[str]
    measurements: Dict[float, float]


def build_series_features(
    formulas: Sequence[str],
    periodic_table: pd.DataFrame,
    miedema: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, stoich_array), with formulas and engineered descriptors in `features`."""
    frame = pd.DataFrame({formula_column: list(formulas)})
    stoich = get_stoich_array(frame, periodic_table, formula_column=formula_column)
    features = add_engineered_features(frame, periodic_table, miedema, formula_column=formula_column)
    return features, stoich


def build_series_panel(
    series: AlloySeries,
    feature_columns: Sequence[str],
    models: Mapping[str, Any],
    periodic_table: pd.DataFrame,
    miedema: pd.DataFrame,
) -> ComparisonPanel:
    """Predict an alloy series with models keyed by display name and pair it with measurements.

    `feature_columns` follows training order; `periodic_table` is indexed by element symbol and
    `miedema` is a symmetric element-pair matrix.
    """
    features, stoich_array = build_series_features(series.formulas, periodic_table, miedema)
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
    miedema: pd.DataFrame,
) -> List[ComparisonPanel]:
    """Build one prediction-and-measurement panel per alloy series."""
    if not models:
        raise ValueError("A case study needs at least one fitted model.")
    return [build_series_panel(series, feature_columns, models, periodic_table, miedema) for series in all_series]


def plot_interpretation(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    explain_models: Mapping[str, Any],
    case_study_models: Mapping[str, Any],
    case_studies: Sequence[AlloySeries],
    target_column: str,
    plots_dir: Path,
    prefix: str = "",
    periodic_table_path: Optional[str] = None,
    miedema_path: Optional[str] = None,
    random_state: int = 0,
) -> None:
    """Save explanation figures for `explain_models` and a case-study figure for `case_study_models`.

    Both map display names to fitted estimators; an empty mapping skips that figure. Explanations use the
    validation rows with `X_train` as the SHAP background. Case studies are featurized with the element
    tables at the given paths (packaged tables when omitted) and saved as `<prefix>case_studies.png`.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if explain_models:
        plot_model_explanations(
            explain_models,
            X_train,
            X_valid,
            y_valid,
            save_dir=plots_dir,
            prefix=prefix,
            random_state=random_state,
        )

    if case_study_models:
        periodic_table, miedema = load_element_properties(periodic_table_path, miedema_path)
        panels = build_series_panels(case_studies, list(X_train.columns), case_study_models, periodic_table, miedema)
        plot_predictions_against_measurements(
            panels,
            y_label=target_column,
            save_path=plots_dir / f"{prefix}case_studies.png",
        )
