"""Run the magnetism pipeline for Novamag or Materials Project data."""

import argparse
import yaml

from data import load_features_and_target, load_raw_data, split_dataset
from preprocess_data import load_elemental_data, load_novamag_raw_data, load_mp_raw_data

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
        # default="./configs/novamag.yaml",
        default="./configs/mp.yaml",
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
    data_visualization = cfg["enable_data_visualization"]
    hyperparameter_tuning = cfg["enable_hyperparameter_tuning"]
    models = cfg["models"]
    
    if dataset_name.lower() == "novamag":
        prefix = "novamag"
    elif dataset_name.lower() == "mp":
        prefix = "mp"
    else:
        raise ValueError("Invalid dataset name. Choose either 'Novamag' or 'Materials Project'.")

    # 0) Load data and elemental tables
    X, y, feature_columns = load_features_and_target(dataset_path)
    pt, mm = load_elemental_data(pt_path, mm_path)

    # 1) Train/validation split
    X_train, X_valid, y_train, y_valid = split_dataset(X, y, train_size=0.8)

    # 2) Train models (optionally tuned)
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

    # 3) Report validation metrics
    print_regression_results(y_valid, preds)

    # 4) Data visualization
    if data_visualization:
        X_raw = load_raw_data(dataset_path)
        plot_ms_distribution_by_tm(X_raw, save_path=plots_save_dir + f"{prefix}_ms_distribution_by_tm.png")
        plot_violin_ms_by_tm(X_raw, save_path=plots_save_dir + f"{prefix}_violin_ms_by_tm.png")
        summarize_compound_radix(X_raw, pt)

    # 5) Permutation importance
    if "rf" in trained_models:
        plot_permutation_importance(trained_models["rf"], X_valid, y_valid, title=f"RF Permutation Importance ({prefix})", 
                                    save_path=plots_save_dir + f"{prefix}_perm_importance_rf.png")

    # 6) SHAP summary
    if "rf" in trained_models:
        plot_shap_summary(trained_models["rf"], X_train, X_valid, save_path=plots_save_dir + f"{prefix}_shap_summary_rf.png")

    # 7) Case studies
    if "rf" in trained_models and "xgb" in trained_models and "ridge" in trained_models:
        plot_case_studies(feature_columns, trained_models["rf"], trained_models["xgb"], trained_models["ridge"], pt, mm, 
                              save_path=plots_save_dir + f"{prefix}_case_studies.png")


if __name__ == "__main__":
    main()
