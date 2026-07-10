"""FeAl / FeCo / FeCr case studies: model predictions vs. literature measurements."""

from typing import Dict, List

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
) -> pd.DataFrame:
    """Build the full feature matrix for a list of chemical formulas.

    Returns a DataFrame that includes the original "chemical formula" column
    plus all nine engineered features used by the case-study models.
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


def _run_case(
    formulas: List[str],
    literature_ms: Dict[float, float],
    X_cols: List[str],
    rf_model,
    xgb_model,
    ridge_model,
    periodic_table,
    miedema_weight,
):
    """Generate predictions and literature references for one case study (no plotting)."""
    X, stoich_array = _build_case_features(formulas, periodic_table, miedema_weight)

    rf_preds = rf_model.predict(X[X_cols])
    xgb_preds = xgb_model.predict(X[X_cols])
    ridge_preds = ridge_model.predict(X[X_cols])

    at_fraction = alloy_transform.get_atomic_frac(stoich_array)
    exp = pd.Series(literature_ms)

    return at_fraction, rf_preds, xgb_preds, ridge_preds, exp


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


def _plot_one_case(ax, at_fraction, element_col, rf_preds, xgb_preds, ridge_preds, exp, title):
    """Plot one case study's predictions + literature scatter onto `ax`."""
    sns.scatterplot(x=at_fraction[element_col], y=rf_preds, ax=ax)
    sns.scatterplot(x=at_fraction[element_col], y=xgb_preds, ax=ax)
    sns.scatterplot(x=at_fraction[element_col], y=ridge_preds, ax=ax)
    sns.scatterplot(x=exp.index, y=exp.values, ax=ax)
    ax.set_title(f"{title} Case Study", fontsize=16)
    ax.set_xlabel(f"{element_col} content [atomic fraction]", fontsize=16)
    ax.set_ylabel("Saturation Magnetisation [T]", fontsize=16)
    legend = ax.legend(
        ["random forest", "xgboost", "ridge regression", "literature"],
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
        at_fraction, rf_preds, xgb_preds, ridge_preds, exp = case_fn(
            feature_columns, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight
        )
        _plot_one_case(ax, at_fraction, element_col, rf_preds, xgb_preds, ridge_preds, exp, title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
