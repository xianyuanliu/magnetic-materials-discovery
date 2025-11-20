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
    split_dataset,
    load_periodic_tables,
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
    # ====== 1. Load Novamag data and split ======
    X_nv, y_nv, my_cols, PT, MM = load_novamag_dataset()
    X_train, X_valid, y_train, y_valid = split_dataset(X_nv, y_nv, train_size=0.8)

    # Only retain the feature columns as a safeguard
    X_train = X_train[my_cols].copy()
    X_valid = X_valid[my_cols].copy()

    # ====== 2. Train the three models on Novamag ======
    rf_model = train_rf(X_train, y_train)
    xgb_model = train_xgb(X_train, y_train)
    ridge_model = train_ridge(X_train, y_train)

    # ====== 3. Predict on Novamag ======
    rfpreds = rf_model.predict(X_valid)
    xgbpreds = xgb_model.predict(X_valid)
    ridgepreds = ridge_model.predict(X_valid)

    # # ====== 4. Load the MP data and split ======
    # X_mp, mp_y, mp_cols, PT_mp, MM_mp = load_mp_dataset()
    # mp_X_train, mp_X_valid, mp_y_train, mp_y_valid = split_dataset(X_mp, mp_y, 0.8)

    # mp_X_train = mp_X_train[mp_cols].copy()
    # mp_X_valid = mp_X_valid[mp_cols].copy()

    # # ====== 5. Retrain and predict on MP using the same models (matches the notebook) ======
    # rf_model.fit(mp_X_train, mp_y_train)
    # xgb_model.fit(mp_X_train, mp_y_train)
    # ridge_model.fit(mp_X_train, mp_y_train)

    # mp_rfpreds = rf_model.predict(mp_X_valid)
    # mp_xgbpreds = xgb_model.predict(mp_X_valid)
    # mp_ridgepreds = ridge_model.predict(mp_X_valid)

    # ====== 6. Print all metrics (notebook cells 96/98/100) ======
    preds = {
    "Random Forest": rfpreds,
    "XGBoost": xgbpreds,
    "Ridge": ridgepreds,
    }

    print_regression_results(y_valid, preds)


    
    # print_all_metrics(
    #     y_valid,
    #     rfpreds,
    #     xgbpreds,
    #     ridgepreds,
    #     # mp_y_valid,
    #     # mp_rfpreds,
    #     # mp_xgbpreds,
    #     # mp_ridgepreds,
    # )

    # ====== 7. Plot Novamag RF permutation importance ======
    plot_permutation_importance(
        rf_model, X_valid, y_valid, title="RF Permutation Importance (Novamag)"
    )

    # ====== 8. Plot the three Novamag case-study panels ======
    plot_novamag_case_studies(my_cols, rf_model, xgb_model, ridge_model, PT, MM)

    # # ====== 9. MP FeAl case study (compute data only, no plots) ======
    # mp_at_FeAl_fraction, mp_rfpreds_FeAl, mp_xgbpreds_FeAl, mp_ridgepreds_FeAl, mp_Exp_FeAl = mp_feal_case(
    #     mp_cols, rf_model, xgb_model, ridge_model, PT_mp, MM_mp
    # )
    # print("MP FeAl case study predictions (RF):", mp_rfpreds_FeAl)


if __name__ == "__main__":
    main()
