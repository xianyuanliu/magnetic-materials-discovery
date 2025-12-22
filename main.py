"""
Run the magnetism pipeline for Novamag or Materials Project data:
- Load and clean the chosen dataset
- Engineer alloy features and split into train/validation
- Train machine learning models
- Report metrics and generate plots (distributions, SHAP, permutation importance, case studies)
"""

import argparse
import yaml

from data import (
    load_mp_raw_data,
    load_novamag_raw_data,
    process_data,
    split_dataset,
)
from train import MODEL_REGISTRY

from evaluate import (
    print_regression_results,
    plot_permutation_importance,
    plot_shap_summary,
    plot_case_studies,
)
from visualize import plot_ms_distribution_by_tm, plot_violin_ms_by_tm, summarize_compound_radix

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ML models for material property prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/novamag.yaml",
        # default="./configs/mp.yaml",
        help="Path to YAML configuration file"
    )
    return parser.parse_args()

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    pt_path = "./data/Periodic-table/periodic_table.xlsx"
    mm_path = "./data/Miedema-model/Miedema-model-reduced.xlsx"
    plots_save_dir = "./plots/"

    args = parse_args()
    cfg = load_config(args.config)

    dataset_name = cfg["dataset"]
    dataset_path = cfg["dataset_path"]
    data_visualization = cfg["data_visualization"]
    hyperparameter_tuning = cfg["hyperparameter_tuning"]
    models = cfg["models"]
    
    if dataset_name.lower() == "novamag":
        prefix = "novamag"
        X_raw = load_novamag_raw_data(dataset_path)
    elif dataset_name.lower() == "mp":
        prefix = "mp"
        X_raw = load_mp_raw_data(dataset_path)
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
    best_params = {}
    trained_models = {}
    preds = {}

    for key in models:
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {key}")

        model_cfg = MODEL_REGISTRY[key]

        # Check if hyperparameter tuning is needed
        params = None
        if hyperparameter_tuning and model_cfg["tune"] is not None:
            params = model_cfg["tune"](X_train, y_train)

        # Train the model with the tuned or default hyperparameters
        model = model_cfg["train"](X_train, y_train, params=params)
        trained_models[key] = model
        best_params[key] = params
        preds[model_cfg["name"]] = model.predict(X_valid)

    # 6) Report validation metrics
    print_regression_results(y_valid, preds)

    # 7) Plot permutation importance
    if "rf" in trained_models:
        plot_permutation_importance(trained_models["rf"], X_valid, y_valid, title=f"RF Permutation Importance ({prefix})", 
                                    save_path=plots_save_dir + f"{prefix}_perm_importance_rf.png")

    # 8) Plot SHAP summary
    if "rf" in trained_models:
        plot_shap_summary(trained_models["rf"], X_train, X_valid, save_path=plots_save_dir + f"{prefix}_shap_summary_rf.png")

    # 9) Plot case studies
    if "rf" in trained_models and "xgb" in trained_models and "ridge" in trained_models:
        plot_case_studies(feature_columns, trained_models["rf"], trained_models["xgb"], trained_models["ridge"], pt, mm, 
                              save_path=plots_save_dir + f"{prefix}_case_studies.png")


if __name__ == "__main__":
    main()
