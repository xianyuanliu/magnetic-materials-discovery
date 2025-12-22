"""
Run the magnetism pipeline for Novamag or Materials Project data:
- Load and clean the chosen dataset
- Engineer alloy features and split into train/validation
- Train machine learning models
- Report metrics and generate plots (distributions, SHAP, permutation importance, case studies)
"""

import argparse

from data import (
    load_mp_raw_data,
    load_novamag_raw_data,
    process_data,
    split_dataset,
)
from train import (
    MODEL_REGISTRY,
    train_linear_regression, train_ridge, train_lasso, train_elasticnet,
    train_rf, train_xgb,
    train_svr, train_mlp,
    tune_ridge_hyperparams, tune_lasso_hyperparams, tune_elasticnet_hyperparams,
    tune_rf_hyperparams, tune_xgb_hyperparams,
    tune_svr_hyperparams, tune_mlp_hyperparams,
)

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

    dataset_name = "Novamag"
    # dataset_name = "MP"

    models = ["linear", "ridge", "lasso", "elasticnet", "rf", "xgb", "svr", "mlp"]

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
    # if hyperparameter_tuning:
    #     ridge_best_params = tune_ridge_hyperparams(X_train, y_train)
    #     lasso_best_params = tune_lasso_hyperparams(X_train, y_train)
    #     elasticnet_best_params = tune_elasticnet_hyperparams(X_train, y_train)
    #     rf_best_params = tune_rf_hyperparams(X_train, y_train)
    #     xgb_best_params = tune_xgb_hyperparams(X_train, y_train)
    #     svr_best_params = tune_svr_hyperparams(X_train, y_train)
    #     mlp_best_params = tune_mlp_hyperparams(X_train, y_train)
    # else:
    #     ridge_best_params = None
    #     lasso_best_params = None
    #     elasticnet_best_params = None
    #     rf_best_params = None
    #     xgb_best_params = None
    #     svr_best_params = None
    #     mlp_best_params = None

    best_params = {}
    trained_models = {}
    preds = {}

    for key in models:
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {key}")

        model_cfg = MODEL_REGISTRY[key]

        params = None
        if hyperparameter_tuning and model_cfg["tune"] is not None:
            params = model_cfg["tune"](X_train, y_train)

        model = model_cfg["train"](X_train, y_train, params=params)
        trained_models[key] = model
        best_params[key] = params
        preds[model_cfg["name"]] = model.predict(X_valid)

    print_regression_results(y_valid, preds)

    # linear_model = train_linear_regression(X_train, y_train) 
    # ridge_model = train_ridge(X_train, y_train, params=ridge_best_params)
    # lasso_model = train_lasso(X_train, y_train, params=lasso_best_params)
    # elasticnet_model = train_elasticnet(X_train, y_train, params=elasticnet_best_params)
    # rf_model = train_rf(X_train, y_train, params=rf_best_params)
    # xgb_model = train_xgb(X_train, y_train, params=xgb_best_params)
    # svr_model = train_svr(X_train, y_train, params=svr_best_params)
    # mlp_model = train_mlp(X_train, y_train, params=mlp_best_params)

    # # 6) Predict on the validation set
    # linear_preds = linear_model.predict(X_valid)
    # ridge_preds = ridge_model.predict(X_valid)
    # lasso_preds = lasso_model.predict(X_valid)
    # elasticnet_preds = elasticnet_model.predict(X_valid)
    # rf_preds = rf_model.predict(X_valid)
    # xgb_preds = xgb_model.predict(X_valid)
    # svr_preds = svr_model.predict(X_valid)
    # mlp_preds = mlp_model.predict(X_valid)

    # # 7) Report validation metrics
    # preds = {
    #     "Linear Regression": linear_preds, "Ridge": ridge_preds, "Lasso": lasso_preds, "ElasticNet": elasticnet_preds,
    #     "Random Forest": rf_preds, "XGBoost": xgb_preds, 
    #     "SVR": svr_preds, "MLP": mlp_preds}
    # print_regression_results(y_valid, preds)

    # 8) Plot permutation importance
    if "rf" in trained_models:
        plot_permutation_importance(trained_models["rf"], X_valid, y_valid, title=f"RF Permutation Importance ({prefix})", 
                                    save_path=plots_save_dir + f"{prefix}_perm_importance_rf.png")

    # 9) Plot SHAP summary
    if "rf" in trained_models:
        plot_shap_summary(trained_models["rf"], X_train, X_valid, save_path=plots_save_dir + f"{prefix}_shap_summary_rf.png")

    # 10) Plot case studies
    if "rf" in trained_models and "xgb" in trained_models and "ridge" in trained_models:
        plot_case_studies(feature_columns, trained_models["rf"], trained_models["xgb"], trained_models["ridge"], pt, mm, 
                              save_path=plots_save_dir + f"{prefix}_case_studies.png")


if __name__ == "__main__":
    main()
