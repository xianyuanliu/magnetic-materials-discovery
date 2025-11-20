# main.py
"""
Main entry point:
- Load the Novamag and Materials Project datasets
- Split into training and validation sets
- Train three models on both datasets
- Print the evaluation metrics
- Plot the key figures (comment out whichever you do not need)
"""

from data import (
    load_novamag_dataset,
    load_mp_dataset,
    load_novamag_raw,
    split_dataset,
)
from train import train_rf, train_ridge, train_xgb
from evaluate import (
    print_regression_results,
    plot_permutation_importance,
    plot_shap_summary,
    plot_novamag_case_studies,
    mp_feal_case,
)


def main():
    novamag_dir = "./data/Novamag_Data_Files/"
    pt_path = "./data/Periodic-table/periodic_table.xlsx"
    mm_path = "./data/Miedema-model/Miedema-model-reduced.xlsx"
    plots_save_dir = "./plots/"

    # ====== 1. Load Novamag data and split ======
    X_raw = load_novamag_raw(novamag_dir)
    X_nv, y_nv, novamag_feature_columns, pt, mm = load_novamag_dataset(X_raw, pt_path, mm_path)

    # Split dataset
    X_train, X_valid, y_train, y_valid = split_dataset(X_nv, y_nv, train_size=0.8)

    # Only retain the feature columns as a safeguard
    X_train = X_train[novamag_feature_columns].copy()
    X_valid = X_valid[novamag_feature_columns].copy()

    # ====== 2. Train the three models on Novamag ======
    rf_model = train_rf(X_train, y_train)
    xgb_model = train_xgb(X_train, y_train)
    ridge_model = train_ridge(X_train, y_train)

    # ====== 3. Predict on Novamag ======
    rfpreds = rf_model.predict(X_valid)
    xgbpreds = xgb_model.predict(X_valid)
    ridgepreds = ridge_model.predict(X_valid)

    # # ====== 4. Load the MP data and split ======
    # X_mp, mp_y, mp_feature_columns, PT_mp, MM_mp = load_mp_dataset()
    # mp_X_train, mp_X_valid, mp_y_train, mp_y_valid = split_dataset(X_mp, mp_y, 0.8)

    # mp_X_train = mp_X_train[mp_feature_columns].copy()
    # mp_X_valid = mp_X_valid[mp_feature_columns].copy()

    # # ====== 5. Retrain and predict on MP using the same models (matches the notebook) ======
    # rf_model.fit(mp_X_train, mp_y_train)
    # xgb_model.fit(mp_X_train, mp_y_train)
    # ridge_model.fit(mp_X_train, mp_y_train)

    # mp_rfpreds = rf_model.predict(mp_X_valid)
    # mp_xgbpreds = xgb_model.predict(mp_X_valid)
    # mp_ridgepreds = ridge_model.predict(mp_X_valid)

    # ====== 6. Print all metrics (notebook cells 96/98/100) ======
    preds = {"Random Forest": rfpreds, "XGBoost": xgbpreds, "Ridge": ridgepreds,}
    print_regression_results(y_valid, preds)

    # ====== 7a. Plot Novamag RF permutation importance ======
    plot_permutation_importance(rf_model, X_valid, y_valid, title="RF Permutation Importance (Novamag)", 
                                save_path=plots_save_dir + "perm_importance_rf_novamag.png")

    # ====== 7b. Plot Novamag RF SHAP summary plot ======
    plot_shap_summary(rf_model, X_train, X_valid, save_path=plots_save_dir + "shap_summary_rf_novamag.png")

    # ====== 8. Plot the three Novamag case-study panels ======
    plot_novamag_case_studies(novamag_feature_columns, rf_model, xgb_model, ridge_model, pt, mm)

    # # ====== 9. MP FeAl case study (compute data only, no plots) ======
    # mp_at_FeAl_fraction, mp_rfpreds_FeAl, mp_xgbpreds_FeAl, mp_ridgepreds_FeAl, mp_Exp_FeAl = mp_feal_case(
    #     mp_feature_columns, rf_model, xgb_model, ridge_model, PT_mp, MM_mp
    # )
    # print("MP FeAl case study predictions (RF):", mp_rfpreds_FeAl)


if __name__ == "__main__":
    main()
