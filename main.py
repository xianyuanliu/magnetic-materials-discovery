# main.py
"""
Run the magnetism pipeline for Novamag or Materials Project data:
- Load and clean the chosen dataset
- Engineer alloy features and split into train/validation
- Train machine learning models
- Report metrics and generate plots (distributions, SHAP, permutation importance, case studies)
"""

from data import (
    load_mp_raw_data,
    load_novamag_raw_data,
    process_data,
    split_dataset,
)
from train import train_rf, train_ridge, train_xgb, tune_rf_hyperparams, tune_ridge_hyperparams, tune_xgb_hyperparams
from evaluate import (
    print_regression_results,
    plot_permutation_importance,
    plot_shap_summary,
    plot_case_studies,
)
from visualize import plot_ms_distribution_by_tm, plot_violin_ms_by_tm, summarize_compound_radix

def main():
    novamag_dir = "./data/Novamag_Data_Files/"
    mp_dir = "./data/mp-data.csv"

    pt_path = "./data/Periodic-table/periodic_table.xlsx"
    mm_path = "./data/Miedema-model/Miedema-model-reduced.xlsx"
    plots_save_dir = "./plots/"
    data_visualization = True
    hyperparameter_tuning = True

    # dataset_name = "Novamag"
    dataset_name = "MP"

    if dataset_name.lower() == "novamag":
        prefix = "novamag"
        X_raw = load_novamag_raw_data(novamag_dir)
    elif dataset_name.lower() == "mp":
        prefix = "mp"
        X_raw = load_mp_raw_data(mp_dir)
    else:
        raise ValueError("Invalid dataset name. Choose either 'Novamag' or 'Materials Project'.")

    # 1) Load, clean, and engineer features for the chosen dataset
    X, y, feature_columns, pt, mm = process_data(X_raw, pt_path, mm_path)

    # 2) Optional raw data visualizations
    if data_visualization:
        plot_ms_distribution_by_tm(X_raw, save_path=plots_save_dir + f"{prefix}_ms_distribution_by_tm.png")
        plot_violin_ms_by_tm(X_raw, save_path=plots_save_dir + f"{prefix}_violin_ms_by_tm.png")
        summarize_compound_radix(X_raw, pt)

    # 3) Split dataset
    X_train, X_valid, y_train, y_valid = split_dataset(X, y, train_size=0.8)

    # 4) Only retain the feature columns as a safeguard
    X_train = X_train[feature_columns].copy()
    X_valid = X_valid[feature_columns].copy()

    # 5) Train the three models (optionally with hyperparameter tuning)
    if hyperparameter_tuning:
        rf_best_params = tune_rf_hyperparams(X_train, y_train)
        xgb_best_params = tune_xgb_hyperparams(X_train, y_train)
        ridge_best_params = tune_ridge_hyperparams(X_train, y_train)
    else:
        rf_best_params = None
        xgb_best_params = None
        ridge_best_params = None

    rf_model = train_rf(X_train, y_train, params=rf_best_params)
    xgb_model = train_xgb(X_train, y_train, params=xgb_best_params)
    ridge_model = train_ridge(X_train, y_train, params=ridge_best_params)

    # 6) Predict on the validation set
    rf_preds = rf_model.predict(X_valid)
    xgb_preds = xgb_model.predict(X_valid)
    ridge_preds = ridge_model.predict(X_valid)

    # 7) Report validation metrics
    preds = {"Random Forest": rf_preds, "XGBoost": xgb_preds, "Ridge": ridge_preds,}
    print_regression_results(y_valid, preds)

    # 8) Plot permutation importance
    plot_permutation_importance(rf_model, X_valid, y_valid, title=f"RF Permutation Importance ({prefix})", 
                                save_path=plots_save_dir + f"{prefix}_perm_importance_rf.png")

    # 9) Plot SHAP summary
    plot_shap_summary(rf_model, X_train, X_valid, save_path=plots_save_dir + f"{prefix}_shap_summary_rf.png")

    # 10) Plot case studies
    plot_case_studies(feature_columns, rf_model, xgb_model, ridge_model, pt, mm, 
                              save_path=plots_save_dir + f"{prefix}_case_studies.png")


if __name__ == "__main__":
    main()
