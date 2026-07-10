"""Model interpretability: permutation importance, SHAP, and literature case studies.

- Permutation feature importance and SHAP summary plots for a trained model.
- FeAl / FeCo / FeCr case studies comparing model predictions against
  literature saturation-magnetization measurements (see case_study_references.py).
"""

from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.inspection import permutation_importance

import alloys as al
from case_study_references import (
    FEAL_FORMULAS,
    FEAL_LITERATURE_MS,
    FECO_FORMULAS,
    FECO_LITERATURE_MS,
    FECR_FORMULAS,
    FECR_LITERATURE_MS,
)

# ====== Permutation Feature Importance & SHAP ======


def plot_permutation_importance(
    model,
    X_valid,
    y_valid,
    title: str = "",
    save_path: str = None,
    random_state: int = 0,
):
    """Plot permutation importance for RFR / XGB / Ridge."""
    perm_import = permutation_importance(
        model, X_valid, y_valid, n_repeats=10, random_state=random_state
    )

    sorted_idx = perm_import.importances_mean.argsort()

    plt.figure(figsize=(14, 7))
    plt.barh(
        range(len(sorted_idx)),
        perm_import.importances_mean[sorted_idx],
        align="center",
    )
    plt.yticks(range(len(sorted_idx)), X_valid.columns[sorted_idx], fontsize=16)
    plt.xlabel("Permutation Feature Importance", fontsize=16)
    plt.ylabel("Features", fontsize=16)
    plt.xticks(fontsize=16)
    if title:
        plt.title(title, fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_shap_summary(model, X_train, X_valid, save_path: str = None):
    """Generate SHAP summary plots."""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_valid, check_additivity=False)

    shap.summary_plot(shap_values, X_valid, feature_names=X_valid.columns, show=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# ====== Case studies: FeAl / FeCo / FeCr ======

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
    stoich = al.get_stoich_array(X, periodic_table)
    X["stoicentw"] = al.get_stoic_entw(stoich)
    X["Zw"] = al.get_zw(periodic_table, stoich)
    X["compoundradix"] = al.get_compound_radix(X)
    X["periodw"] = al.get_periodw(periodic_table, stoich)
    X["groupw"] = al.get_groupw(periodic_table, stoich)
    X["meltingTw"] = al.get_melting_tw(periodic_table, stoich)
    X["miedemaH"] = al.get_miedemaw(miedema_weight, stoich)
    X["valencew"] = al.get_valencew(periodic_table, stoich)
    X["electronegw"] = al.get_electronegw(periodic_table, stoich)
    return X, stoich


def feal_case(X_cols: List[str], rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeAl case study (no plotting)."""
    X_FeAl, stoich_array_FeAl = _build_case_features(FEAL_FORMULAS, periodic_table, miedema_weight)

    rfpreds_FeAl = rf_model.predict(X_FeAl[X_cols])
    xgbpreds_FeAl = xgb_model.predict(X_FeAl[X_cols])
    ridgepreds_FeAl = ridge_model.predict(X_FeAl[X_cols])

    at_FeAl_fraction = al.get_atomic_frac(stoich_array_FeAl)

    Exp_FeAl = pd.Series(FEAL_LITERATURE_MS)

    return at_FeAl_fraction, rfpreds_FeAl, xgbpreds_FeAl, ridgepreds_FeAl, Exp_FeAl


def feco_case(X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeCo case study."""
    X_FeCo, stoich_array_FeCo = _build_case_features(FECO_FORMULAS, periodic_table, miedema_weight)

    rfpreds_FeCo = rf_model.predict(X_FeCo[X_cols])
    xgbpreds_FeCo = xgb_model.predict(X_FeCo[X_cols])
    ridgepreds_FeCo = ridge_model.predict(X_FeCo[X_cols])

    at_FeCo_fraction = al.get_atomic_frac(stoich_array_FeCo)

    Exp_FeCo = pd.Series(FECO_LITERATURE_MS)

    return at_FeCo_fraction, rfpreds_FeCo, xgbpreds_FeCo, ridgepreds_FeCo, Exp_FeCo


def fecr_case(X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeCr case study."""
    X_FeCr, stoich_array_FeCr = _build_case_features(FECR_FORMULAS, periodic_table, miedema_weight)

    rfpreds_FeCr = rf_model.predict(X_FeCr[X_cols])
    xgbpreds_FeCr = xgb_model.predict(X_FeCr[X_cols])
    ridgepreds_FeCr = ridge_model.predict(X_FeCr[X_cols])

    at_FeCr_fraction = al.get_atomic_frac(stoich_array_FeCr)

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
