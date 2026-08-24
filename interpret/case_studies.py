"""FeAl / FeCo / FeCr case studies: model predictions vs. literature measurements."""

from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import prepdata.alloy_transform as alloy_transform
from interpret.case_study_references import (
    FEAL_FORMULAS,
    FEAL_LITERATURE_MS,
    FECO_FORMULAS,
    FECO_LITERATURE_MS,
    FECR_FORMULAS,
    FECR_LITERATURE_MS,
)


def _build_case_features(
    formulas: List[str],
    periodic_table,
    miedema_weight,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full feature matrix for a list of chemical formulas.

    Args:
        formulas: Chemical formulas to featurize.
        periodic_table: Periodic table data, indexed by element symbol.
        miedema_weight: Symmetrised Miedema mixing-enthalpy matrix.

    Returns:
        (features, stoich_array). `features` carries the original
        "chemical formula" column plus all nine engineered features used by
        the case-study models; `stoich_array` is the per-element
        stoichiometry the features were derived from, returned so callers can
        convert it to atomic fractions without re-parsing the formulas.
    """
    X = pd.DataFrame(formulas, columns=["chemical formula"])
    stoich = alloy_transform.get_stoich_array(X, periodic_table)
    X["stoicentw"] = alloy_transform.get_stoic_entw(stoich)
    X["Zw"] = alloy_transform.get_zw(periodic_table, stoich)
    X["compoundradix"] = alloy_transform.get_compound_radix(X)
    X["periodw"] = alloy_transform.get_periodw(periodic_table, stoich)
    X["groupw"] = alloy_transform.get_groupw(periodic_table, stoich)
    X["meltingTw"] = alloy_transform.get_melting_tw(periodic_table, stoich)
    X["miedemaH"] = alloy_transform.get_miedemaw(miedema_weight, stoich)
    X["valencew"] = alloy_transform.get_valencew(periodic_table, stoich)
    X["electronegw"] = alloy_transform.get_electronegw(periodic_table, stoich)
    return X, stoich


class CaseStudyResult(NamedTuple):
    """Model predictions and literature values for one binary-alloy case study.

    Attributes:
        atomic_fraction: Per-element atomic fractions, one row per formula.
        predictions: Predicted saturation magnetization per model name, in the
            order the models were passed in.
        literature: Measured saturation magnetization, indexed by the alloying
            element's atomic fraction.
    """

    atomic_fraction: pd.DataFrame
    predictions: Dict[str, np.ndarray]
    literature: pd.Series


def _run_case(
    formulas: List[str],
    literature_ms: Dict[float, float],
    X_cols: List[str],
    rf_model,
    xgb_model,
    ridge_model,
    periodic_table,
    miedema_weight,
) -> CaseStudyResult:
    """Generate predictions and literature references for one case study (no plotting).

    Args:
        formulas: Chemical formulas spanning the alloy series.
        literature_ms: Measured saturation magnetization, keyed by the
            alloying element's atomic fraction.
        X_cols: Feature columns the models were trained on, in training order.
        rf_model: Fitted random forest.
        xgb_model: Fitted XGBoost regressor.
        ridge_model: Fitted ridge regressor.
        periodic_table: Periodic table data, indexed by element symbol.
        miedema_weight: Symmetrised Miedema mixing-enthalpy matrix.

    Returns:
        A CaseStudyResult holding the atomic fractions, one prediction array
        per model, and the literature series.
    """
    features, stoich_array = _build_case_features(formulas, periodic_table, miedema_weight)

    return CaseStudyResult(
        atomic_fraction=alloy_transform.get_atomic_frac(stoich_array),
        predictions={
            "random forest": rf_model.predict(features[X_cols]),
            "xgboost": xgb_model.predict(features[X_cols]),
            "ridge regression": ridge_model.predict(features[X_cols]),
        },
        literature=pd.Series(literature_ms),
    )


def feal_case(X_cols: List[str], rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeAl case study (no plotting)."""
    return _run_case(
        FEAL_FORMULAS, FEAL_LITERATURE_MS, X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight
    )


def feco_case(X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeCo case study."""
    return _run_case(
        FECO_FORMULAS, FECO_LITERATURE_MS, X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight
    )


def fecr_case(X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeCr case study."""
    return _run_case(
        FECR_FORMULAS, FECR_LITERATURE_MS, X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight
    )


def _plot_one_case(ax, result: CaseStudyResult, element_col: str, title: str):
    """Plot one case study's predictions + literature scatter onto `ax`.

    The legend is built from `result.predictions` rather than hard-coded, so
    it stays correct if the set of compared models changes.
    """
    for y_pred in result.predictions.values():
        sns.scatterplot(x=result.atomic_fraction[element_col], y=y_pred, ax=ax)
    sns.scatterplot(x=result.literature.index, y=result.literature.values, ax=ax)
    ax.set_title(f"{title} Case Study", fontsize=16)
    ax.set_xlabel(f"{element_col} content [atomic fraction]", fontsize=16)
    ax.set_ylabel("Saturation Magnetisation [T]", fontsize=16)
    legend = ax.legend(
        [*result.predictions, "literature"],
        loc="upper right",
        fontsize=12,
    )
    legend.get_frame().set_facecolor("white")


def plot_case_studies(
    feature_columns,
    rf_model,
    xgb_model,
    ridge_model,
    periodic_table,
    miedema_weight,
    save_path=None,
):
    """Plot three case studies (FeAl, FeCo, FeCr) side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    cases = [
        (feal_case, "Al", "FeAl", axes[0]),
        (feco_case, "Co", "FeCo", axes[1]),
        (fecr_case, "Cr", "FeCr", axes[2]),
    ]

    for case_fn, element_col, title, ax in cases:
        result = case_fn(
            feature_columns, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight
        )
        _plot_one_case(ax, result, element_col, title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
