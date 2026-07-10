"""FeAl / FeCo / FeCr case studies: model predictions vs. literature measurements."""

from typing import List

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


def feal_case(X_cols: List[str], rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeAl case study (no plotting)."""
    X_FeAl, stoich_array_FeAl = _build_case_features(FEAL_FORMULAS, periodic_table, miedema_weight)

    rfpreds_FeAl = rf_model.predict(X_FeAl[X_cols])
    xgbpreds_FeAl = xgb_model.predict(X_FeAl[X_cols])
    ridgepreds_FeAl = ridge_model.predict(X_FeAl[X_cols])

    at_FeAl_fraction = alloy_transform.get_atomic_frac(stoich_array_FeAl)

    Exp_FeAl = pd.Series(FEAL_LITERATURE_MS)

    return at_FeAl_fraction, rfpreds_FeAl, xgbpreds_FeAl, ridgepreds_FeAl, Exp_FeAl


def feco_case(X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeCo case study."""
    X_FeCo, stoich_array_FeCo = _build_case_features(FECO_FORMULAS, periodic_table, miedema_weight)

    rfpreds_FeCo = rf_model.predict(X_FeCo[X_cols])
    xgbpreds_FeCo = xgb_model.predict(X_FeCo[X_cols])
    ridgepreds_FeCo = ridge_model.predict(X_FeCo[X_cols])

    at_FeCo_fraction = alloy_transform.get_atomic_frac(stoich_array_FeCo)

    Exp_FeCo = pd.Series(FECO_LITERATURE_MS)

    return at_FeCo_fraction, rfpreds_FeCo, xgbpreds_FeCo, ridgepreds_FeCo, Exp_FeCo


def fecr_case(X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeCr case study."""
    X_FeCr, stoich_array_FeCr = _build_case_features(FECR_FORMULAS, periodic_table, miedema_weight)

    rfpreds_FeCr = rf_model.predict(X_FeCr[X_cols])
    xgbpreds_FeCr = xgb_model.predict(X_FeCr[X_cols])
    ridgepreds_FeCr = ridge_model.predict(X_FeCr[X_cols])

    at_FeCr_fraction = alloy_transform.get_atomic_frac(stoich_array_FeCr)

    Exp_FeCr = pd.Series(FECR_LITERATURE_MS)

    return at_FeCr_fraction, rfpreds_FeCr, xgbpreds_FeCr, ridgepreds_FeCr, Exp_FeCr


def plot_case_studies(
    feature_columns,
    rf_model,
    xgb_model,
    ridge_model,
    periodic_table,
    miedema_weight,
    save_path = None,
):
    """Plot three case studies (FeAl, FeCo, FeCr) side by side."""
    (
        at_FeAl_fraction,
        rfpreds_FeAl,
        xgbpreds_FeAl,
        ridgepreds_FeAl,
        Exp_FeAl,
    ) = feal_case(feature_columns, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight)

    (
        at_FeCo_fraction,
        rfpreds_FeCo,
        xgbpreds_FeCo,
        ridgepreds_FeCo,
        Exp_FeCo,
    ) = feco_case(feature_columns, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight)

    (
        at_FeCr_fraction,
        rfpreds_FeCr,
        xgbpreds_FeCr,
        ridgepreds_FeCr,
        Exp_FeCr,
    ) = fecr_case(feature_columns, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

    # FeAl
    sns.scatterplot(x=at_FeAl_fraction["Al"], y=rfpreds_FeAl, ax=ax1)
    sns.scatterplot(x=at_FeAl_fraction["Al"], y=xgbpreds_FeAl, ax=ax1)
    sns.scatterplot(x=at_FeAl_fraction["Al"], y=ridgepreds_FeAl, ax=ax1)
    sns.scatterplot(x=Exp_FeAl.index, y=Exp_FeAl.values, ax=ax1)
    ax1.set_title("FeAl Case Study", fontsize=16)
    ax1.set_xlabel("Al content [atomic fraction]", fontsize=16)
    ax1.set_ylabel("Saturation Magnetisation [T]", fontsize=16)
    legend1 = ax1.legend(
        ["random forest", "xgboost", "ridge regression", "literature"],
        loc="upper right",
        fontsize=12,
    )
    legend1.get_frame().set_facecolor("white")

    # FeCo
    sns.scatterplot(x=at_FeCo_fraction["Co"], y=rfpreds_FeCo, ax=ax2)
    sns.scatterplot(x=at_FeCo_fraction["Co"], y=xgbpreds_FeCo, ax=ax2)
    sns.scatterplot(x=at_FeCo_fraction["Co"], y=ridgepreds_FeCo, ax=ax2)
    sns.scatterplot(x=Exp_FeCo.index, y=Exp_FeCo.values, ax=ax2)
    ax2.set_title("FeCo Case Study", fontsize=16)
    ax2.set_xlabel("Co content [atomic fraction]", fontsize=16)
    ax2.set_ylabel("Saturation Magnetisation [T]", fontsize=16)
    legend2 = ax2.legend(
        ["random forest", "xgboost", "ridge regression", "literature"],
        loc="upper right",
        fontsize=12,
    )
    legend2.get_frame().set_facecolor("white")

    # FeCr
    sns.scatterplot(x=at_FeCr_fraction["Cr"], y=rfpreds_FeCr, ax=ax3)
    sns.scatterplot(x=at_FeCr_fraction["Cr"], y=xgbpreds_FeCr, ax=ax3)
    sns.scatterplot(x=at_FeCr_fraction["Cr"], y=ridgepreds_FeCr, ax=ax3)
    sns.scatterplot(x=Exp_FeCr.index, y=Exp_FeCr.values, ax=ax3)
    ax3.set_title("FeCr Case Study", fontsize=16)
    ax3.set_xlabel("Cr content [atomic fraction]", fontsize=16)
    ax3.set_ylabel("Saturation Magnetisation [T]", fontsize=16)

    legend3 = ax3.legend(
        ["random forest", "xgboost", "ridge regression", "literature"],
        loc="upper right",
        fontsize=12,
    )
    legend3.get_frame().set_facecolor("white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
