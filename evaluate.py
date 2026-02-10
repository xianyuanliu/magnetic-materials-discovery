"""
Evaluation and visualization:
- Quantitative metrics (MSE / MAE / R²)
- Permutation importance
- SHAP
- FeAl / FeCo / FeCr case studies
- Optional dataset distributions - Ms histograms and violin plots
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import alloys as al


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_keys: List[str],
    model_registry: Dict,
    hyperparameter_tuning: bool = False,
    cv_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
):
    """Run K-fold cross-validation for the requested models."""
    results = {}
    for key in model_keys:
        if key not in model_registry:
            raise ValueError(f"Unknown model key: {key}")
        name = model_registry[key]["name"]
        results[name] = {"mse": [], "mae": [], "r2": []}

    kf = KFold(
        n_splits=cv_folds,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )

    # --- Fold-by-fold RF vs XGB tracking (minimal addition) ---
    rf_fold_mse = []
    xgb_fold_mse = []

    for fold_id, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        for key in model_keys:
            model_cfg = model_registry[key]
            params = None
            if hyperparameter_tuning and model_cfg["tune"] is not None:
                params = model_cfg["tune"](X_train, y_train)

            model = model_cfg["train"](X_train, y_train, params=params)
            y_pred = model.predict(X_valid)

            mse = mean_squared_error(y_valid, y_pred)
            mae = mean_absolute_error(y_valid, y_pred)
            r2 = r2_score(y_valid, y_pred)

            name = model_cfg["name"]
            results[name]["mse"].append(mse)
            results[name]["mae"].append(mae)
            results[name]["r2"].append(r2)

            # record fold MSE for RF/XGB only
            if key == "rf":
                rf_fold_mse.append(mse)
            elif key == "xgb":
                xgb_fold_mse.append(mse)

    # --- Fold-by-fold comparison RF vs XGB (supervisor request) ---
    if len(rf_fold_mse) == cv_folds and len(xgb_fold_mse) == cv_folds:
        wins_rf = 0
        wins_xgb = 0
        ties = 0

        print("\nFold-by-fold comparison (metric=MSE): Random Forest vs XGBoost")
        for i, (m_rf, m_xgb) in enumerate(zip(rf_fold_mse, xgb_fold_mse), start=1):
            if m_rf < m_xgb:
                wins_rf += 1
                winner = "RF"
            elif m_xgb < m_rf:
                wins_xgb += 1
                winner = "XGB"
            else:
                ties += 1
                winner = "Tie"
            print(f"  Fold {i}: RF={m_rf:.6f}  XGB={m_xgb:.6f}  -> {winner}")

        print(f"Summary: RF wins {wins_rf}/{cv_folds}, XGB wins {wins_xgb}/{cv_folds}, ties {ties}")
    else:
        print("\nFold-by-fold RF vs XGB: skipped (need both 'rf' and 'xgb' in models list).")

    return results



# ====== Quantitative metrics ======

def print_holdout_results(y_true, predictions: Dict[str, np.ndarray]):
    """Print MSE, MAE, and R² for multiple regression models."""
    print("Regression Metrics:")
    for name, y_pred in predictions.items():
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2:  {r2:.4f}")

def print_cv_results(results: Dict[str, Dict[str, List[float]]]):
    """Print mean ± std metrics for cross-validation results."""
    print("Cross-Validation Metrics (mean ± std):")
    for name, scores in results.items():
        mse_mean = np.mean(scores["mse"])
        mse_std = np.std(scores["mse"])
        mae_mean = np.mean(scores["mae"])
        mae_std = np.std(scores["mae"])
        r2_mean = np.mean(scores["r2"])
        r2_std = np.std(scores["r2"])

        print(f"\n{name}:")
        print(f"  MSE: {mse_mean:.4f} ± {mse_std:.4f}")
        print(f"  MAE: {mae_mean:.4f} ± {mae_std:.4f}")
        print(f"  R2:  {r2_mean:.4f} ± {r2_std:.4f}")

# ====== Permutation Feature Importance & SHAP ======

def plot_permutation_importance(model, X_valid, y_valid, title: str = "", save_path: str = None):
    """Plot permutation importance for RFR / XGB / Ridge."""
    perm_import = permutation_importance(
        model, X_valid, y_valid, n_repeats=10, random_state=0
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

def feal_case(X_cols: List[str], rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeAl case study (no plotting)."""
    # Chemical formulas we want predictions for
    X_FeAl = pd.DataFrame(
        [
            "Fe93Al2",
            "Fe96Al4",
            "Fe94Al6",
            "Fe93Al7",
            "Fe90Al10",
            "Fe88Al12",
            "Fe85Al15",
            "Fe82Al18",
            "Fe78Al22",
            "Fe75Al25",
            "Fe72Al28",
            "Fe68Al32",
            "Fe66Al34",
        ],
        columns=["chemical formula"],
    )

    stoich_array_FeAl = al.get_stoich_array(X_FeAl, periodic_table)

    X_FeAl["stoicentw"] = al.get_StoicEntw(stoich_array_FeAl)
    X_FeAl["Zw"] = al.get_Zw(periodic_table, stoich_array_FeAl)
    X_FeAl["compoundradix"] = al.get_CompoundRadix(X_FeAl)
    X_FeAl["periodw"] = al.get_Periodw(periodic_table, stoich_array_FeAl)
    X_FeAl["groupw"] = al.get_Groupw(periodic_table, stoich_array_FeAl)
    X_FeAl["meltingTw"] = al.get_MeltingTw(periodic_table, stoich_array_FeAl)
    X_FeAl["miedemaH"] = al.get_Miedemaw(miedema_weight, stoich_array_FeAl)
    X_FeAl["valencew"] = al.get_Valencew(periodic_table, stoich_array_FeAl)
    X_FeAl["electronegw"] = al.get_Electronegw(periodic_table, stoich_array_FeAl)

    rfpreds_FeAl = rf_model.predict(X_FeAl[X_cols])
    xgbpreds_FeAl = xgb_model.predict(X_FeAl[X_cols])
    ridgepreds_FeAl = ridge_model.predict(X_FeAl[X_cols])

    at_FeAl_fraction = al.get_AtomicFrac(stoich_array_FeAl)

    Exp_FeAl = pd.Series(
        data=[
            2.14,
            2.12,
            2.09,
            2.05,
            2.01,
            1.98,
            1.92,
            1.86,
            1.80,
            1.75,
            1.69,
            1.65,
            1.60,
        ],
        index=[
            0.02,
            0.038,
            0.058,
            0.074,
            0.099,
            0.124,
            0.152,
            0.183,
            0.219,
            0.251,
            0.281,
            0.315,
            0.341,
        ],
    )

    return at_FeAl_fraction, rfpreds_FeAl, xgbpreds_FeAl, ridgepreds_FeAl, Exp_FeAl


def feco_case(X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeCo case study."""
    X_FeCo = pd.DataFrame(
        [
            "Fe100Co0",
            "Fe96Co4",
            "Fe92Co8",
            "Fe90Co10",
            "Fe88Co12",
            "Fe85Co15",
            "Fe82Co18",
            "Fe79Co21",
            "Fe71Co29",
            "Fe59Co41",
            "Fe45Co55",
            "Fe26Co74",
            "Fe7Co93",
        ],
        columns=["chemical formula"],
    )

    stoich_array_FeCo = al.get_stoich_array(X_FeCo, periodic_table)

    X_FeCo["stoicentw"] = al.get_StoicEntw(stoich_array_FeCo)
    X_FeCo["Zw"] = al.get_Zw(periodic_table, stoich_array_FeCo)
    X_FeCo["compoundradix"] = al.get_CompoundRadix(X_FeCo)
    X_FeCo["periodw"] = al.get_Periodw(periodic_table, stoich_array_FeCo)
    X_FeCo["groupw"] = al.get_Groupw(periodic_table, stoich_array_FeCo)
    X_FeCo["meltingTw"] = al.get_MeltingTw(periodic_table, stoich_array_FeCo)
    X_FeCo["miedemaH"] = al.get_Miedemaw(miedema_weight, stoich_array_FeCo)
    X_FeCo["valencew"] = al.get_Valencew(periodic_table, stoich_array_FeCo)
    X_FeCo["electronegw"] = al.get_Electronegw(periodic_table, stoich_array_FeCo)
    rfpreds_FeCo = rf_model.predict(X_FeCo[X_cols])
    xgbpreds_FeCo = xgb_model.predict(X_FeCo[X_cols])
    ridgepreds_FeCo = ridge_model.predict(X_FeCo[X_cols])

    at_FeCo_fraction = al.get_AtomicFrac(stoich_array_FeCo)

    Exp_FeCo = pd.Series(
        data=[
            2.18,
            2.21,
            2.24,
            2.26,
            2.30,
            2.33,
            2.36,
            2.39,
            2.43,
            2.44,
            2.31,
            2.09,
            1.8,
        ],
        index=[0.00, 0.04, 0.08, 0.10, 0.12, 0.15, 0.18, 0.21, 0.29, 0.41, 0.55, 0.74, 0.93],
    )

    return at_FeCo_fraction, rfpreds_FeCo, xgbpreds_FeCo, ridgepreds_FeCo, Exp_FeCo


def fecr_case(X_cols, rf_model, xgb_model, ridge_model, periodic_table, miedema_weight):
    """Generate predictions and literature references for the FeCr case study."""
    X_FeCr = pd.DataFrame(
        [
            "Fe99Cr1",
            "Fe98Cr2",
            "Fe96Cr4",
            "Fe95Cr5",
            "Fe93Cr7",
            "Fe92Cr8",
            "Fe91Cr9",
            "Fe90Cr10",
            "Fe89Cr11",
            "Fe87Cr13",
            "Fe85Cr15",
            "Fe83Cr17",
            "Fe80Cr20",
        ],
        columns=["chemical formula"],
    )

    stoich_array_FeCr = al.get_stoich_array(X_FeCr, periodic_table)

    X_FeCr["stoicentw"] = al.get_StoicEntw(stoich_array_FeCr)
    X_FeCr["Zw"] = al.get_Zw(periodic_table, stoich_array_FeCr)
    X_FeCr["compoundradix"] = al.get_CompoundRadix(X_FeCr)
    X_FeCr["periodw"] = al.get_Periodw(periodic_table, stoich_array_FeCr)
    X_FeCr["groupw"] = al.get_Groupw(periodic_table, stoich_array_FeCr)
    X_FeCr["meltingTw"] = al.get_MeltingTw(periodic_table, stoich_array_FeCr)
    X_FeCr["miedemaH"] = al.get_Miedemaw(miedema_weight, stoich_array_FeCr)
    X_FeCr["valencew"] = al.get_Valencew(periodic_table, stoich_array_FeCr)
    X_FeCr["electronegw"] = al.get_Electronegw(periodic_table, stoich_array_FeCr)

    rfpreds_FeCr = rf_model.predict(X_FeCr[X_cols])
    xgbpreds_FeCr = xgb_model.predict(X_FeCr[X_cols])
    ridgepreds_FeCr = ridge_model.predict(X_FeCr[X_cols])

    at_FeCr_fraction = al.get_AtomicFrac(stoich_array_FeCr)

    Exp_FeCr = pd.Series(
        data=[2.14, 2.095, 2.00, 1.96, 1.92, 1.89, 1.86, 1.83, 1.78, 1.73, 1.66, 1.60],
        index=[0.01, 0.02, 0.04, 0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.15, 0.17],
    )

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

    # FeCo
    sns.scatterplot(x=at_FeCo_fraction["Co"], y=rfpreds_FeCo, ax=ax2)
    sns.scatterplot(x=at_FeCo_fraction["Co"], y=xgbpreds_FeCo, ax=ax2)
    sns.scatterplot(x=at_FeCo_fraction["Co"], y=ridgepreds_FeCo, ax=ax2)
    sns.scatterplot(x=Exp_FeCo.index, y=Exp_FeCo.values, ax=ax2)
    ax2.set_title("FeCo Case Study", fontsize=16)
    ax2.set_xlabel("Co content [atomic fraction]", fontsize=16)
    ax2.set_ylabel("Saturation Magnetisation [T]", fontsize=16)

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
