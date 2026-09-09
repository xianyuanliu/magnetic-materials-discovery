"""Binary-alloy case studies: model predictions vs. literature measurements.

The comparison is over whatever models the caller passes in, keyed by name, so
adding or dropping a model is a config change rather than an edit here — it used
to take exactly a random forest, an XGBoost and a ridge, in that order.
"""

from typing import Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.persistence import align_features
from prepdata import alloy_transform
from prepdata.build_features import add_engineered_features
from interpret.case_study_references import (
    FEAL_FORMULAS,
    FEAL_LITERATURE_MS,
    FECO_FORMULAS,
    FECO_LITERATURE_MS,
    FECR_FORMULAS,
    FECR_LITERATURE_MS,
)


class CaseStudy(NamedTuple):
    """One alloy series to compare against literature.

    Attributes:
        title: Display name, e.g. "FeCo".
        element: Alloying element whose atomic fraction is the x axis.
        formulas: Chemical formulas spanning the series.
        literature_ms: Measured saturation magnetization, keyed by the alloying
            element's atomic fraction.
    """

    title: str
    element: str
    formulas: List[str]
    literature_ms: Dict[float, float]


# The series plotted by plot_case_studies. Adding one here adds a panel.
CASE_STUDIES: Tuple[CaseStudy, ...] = (
    CaseStudy("FeAl", "Al", FEAL_FORMULAS, FEAL_LITERATURE_MS),
    CaseStudy("FeCo", "Co", FECO_FORMULAS, FECO_LITERATURE_MS),
    CaseStudy("FeCr", "Cr", FECR_FORMULAS, FECR_LITERATURE_MS),
)


class CaseStudyResult(NamedTuple):
    """Model predictions and literature values for one binary-alloy case study.

    Attributes:
        atomic_fraction: Per-element atomic fractions, one row per formula.
        predictions: Predicted target per model name, in the order the models
            were passed in.
        literature: Measured saturation magnetization, indexed by the alloying
            element's atomic fraction.
    """

    atomic_fraction: pd.DataFrame
    predictions: Dict[str, np.ndarray]
    literature: pd.Series


def build_case_features(
    formulas: Sequence[str],
    periodic_table: pd.DataFrame,
    miedema_weight: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature matrix for a list of chemical formulas.

    Shares prepdata.build_features.add_engineered_features with training, so the
    case-study features cannot drift from the ones the models were fitted on.

    Args:
        formulas: Chemical formulas to featurize.
        periodic_table: Periodic table data, indexed by element symbol.
        miedema_weight: Symmetrised Miedema mixing-enthalpy matrix.
        formula_column: Name to give the formula column.

    Returns:
        (features, stoich_array). `features` carries the formula column plus the
        engineered features; `stoich_array` is the per-element stoichiometry
        they were derived from, returned so callers can convert it to atomic
        fractions without re-parsing the formulas.
    """
    frame = pd.DataFrame({formula_column: list(formulas)})
    stoich = alloy_transform.get_stoich_array(frame, periodic_table, formula_column=formula_column)
    features = add_engineered_features(frame, periodic_table, miedema_weight, formula_column=formula_column)
    return features, stoich


def run_case(
    case: CaseStudy,
    feature_columns: Sequence[str],
    models: Mapping[str, Any],
    periodic_table: pd.DataFrame,
    miedema_weight: pd.DataFrame,
) -> CaseStudyResult:
    """Generate predictions and literature references for one case study.

    Args:
        case: The alloy series to evaluate.
        feature_columns: Feature columns the models were trained on, in order.
        models: Fitted models keyed by display name.
        periodic_table: Periodic table data, indexed by element symbol.
        miedema_weight: Symmetrised Miedema mixing-enthalpy matrix.

    Returns:
        A CaseStudyResult holding the atomic fractions, one prediction array per
        model, and the literature series.
    """
    features, stoich_array = build_case_features(case.formulas, periodic_table, miedema_weight)
    X = align_features(features, feature_columns, source=f"the {case.title} case study")

    return CaseStudyResult(
        atomic_fraction=alloy_transform.get_atomic_frac(stoich_array),
        predictions={name: model.predict(X) for name, model in models.items()},
        literature=pd.Series(case.literature_ms),
    )


def _plot_one_case(ax, result: CaseStudyResult, case: CaseStudy, target_label: str) -> None:
    """Plot one case study's predictions and literature scatter onto `ax`.

    The legend is built from `result.predictions` rather than hard-coded, so it
    stays correct whichever models were compared.
    """
    for y_pred in result.predictions.values():
        sns.scatterplot(x=result.atomic_fraction[case.element], y=y_pred, ax=ax)
    sns.scatterplot(x=result.literature.index, y=result.literature.values, ax=ax)

    ax.set_title(f"{case.title} Case Study", fontsize=16)
    ax.set_xlabel(f"{case.element} content [atomic fraction]", fontsize=16)
    ax.set_ylabel(target_label, fontsize=16)
    legend = ax.legend([*result.predictions, "literature"], loc="upper right", fontsize=12)
    legend.get_frame().set_facecolor("white")


def plot_case_studies(
    feature_columns: Sequence[str],
    models: Mapping[str, Any],
    periodic_table: pd.DataFrame,
    miedema_weight: pd.DataFrame,
    cases: Sequence[CaseStudy] = CASE_STUDIES,
    target_label: str = "Saturation Magnetisation [T]",
    save_path=None,
) -> None:
    """Plot every case study side by side, one panel each.

    Args:
        feature_columns: Feature columns the models were trained on, in order.
        models: Fitted models keyed by display name; every one is plotted.
        periodic_table: Periodic table data, indexed by element symbol.
        miedema_weight: Symmetrised Miedema mixing-enthalpy matrix.
        cases: Which alloy series to plot.
        target_label: Y-axis label for the predicted property.
        save_path: If set, save the figure here instead of showing it.
    """
    if not models:
        raise ValueError("plot_case_studies needs at least one fitted model.")

    fig, axes = plt.subplots(1, len(cases), figsize=(6.7 * len(cases), 4), squeeze=False)

    for ax, case in zip(axes[0], cases):
        result = run_case(case, feature_columns, models, periodic_table, miedema_weight)
        _plot_one_case(ax, result, case, target_label)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
