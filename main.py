"""Run the magnetism pipeline for Novamag or Materials Project data."""

import argparse
import yaml
import os

from data import (
    load_features_and_target,
    load_raw_data,
    split_dataset,
    load_train_test_features_and_target,
)
from preprocess_data import load_elemental_data

from train import MODEL_REGISTRY

from evaluate import (
    print_holdout_results,
    cross_validate_models,
    print_cv_results,
    compare_models_significance,
    plot_permutation_importance,
    plot_shap_summary,
    plot_case_studies,
)
from visualize import plot_ms_distribution_by_tm, plot_violin_ms_by_tm, summarize_compound_radix

def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models for material property prediction")
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

    dataset_name = cfg["dataset"].lower()
    dataset_path = cfg.get("dataset_path")
    train_dataset_path = cfg.get("train_dataset_path")
    test_dataset_path = cfg.get("test_dataset_path")
    data_visualization = cfg["enable_data_visualization"]
    hyperparameter_tuning = cfg["enable_hyperparameter_tuning"]
    ablation_study = cfg["enable_ablation_study"]
    models = cfg["models"]
    
    evaluation_mode = cfg["evaluation_mode"].lower()
    cv_folds = cfg["cv_folds"]
    cv_shuffle = cfg["cv_shuffle"]
    cv_random_state = cfg["cv_random_state"]
    cv_seeds = cfg.get("cv_seeds", [cv_random_state])

    if evaluation_mode not in {"holdout", "cross_validation", "ood"}:
        raise ValueError("Invalid evaluation_mode. Choose 'holdout', 'cross_validation', or 'ood'.")
    need_cross_validation = evaluation_mode == "cross_validation"
    need_ood = evaluation_mode == "ood"
    if need_ood and (not train_dataset_path or not test_dataset_path):
        raise ValueError("OOD mode requires train_dataset_path and test_dataset_path in the config.")
    if not need_ood and not dataset_path:
        raise ValueError("dataset_path is required for holdout or cross_validation modes.")
    
    if dataset_name.lower() == "novamag":
        prefix = "novamag"
    elif dataset_name.lower() == "mp":
        prefix = "mp"
    else:
        raise ValueError("Invalid dataset name. Choose either 'Novamag' or 'Materials Project'.")

    # 0) Load data and elemental tables
    best_params = {}
    trained_models = {}
    preds = {}

    
    # 1) Train/validation split 
    if need_cross_validation:
        # run multiple CV runs with different seeds (e.g. 0,5,10,15,20)
        if len(cv_seeds) < 1:
            raise ValueError("cv_seeds must contain at least one seed.")

        # Load once
        X, y, feature_columns = load_features_and_target(dataset_path)

        def _name_for_key(model_key: str) -> str:
            return MODEL_REGISTRY[model_key]["name"]

        rf_name = _name_for_key("rf") if "rf" in models else None
        xgb_name = _name_for_key("xgb") if "xgb" in models else None

        # Run CV for each seed
        for run_i, seed in enumerate(cv_seeds, start=1):
            seed = int(seed)

            print(f"\n==============================")
            print(f"=== CV Run {run_i}/{len(cv_seeds)} (seed={seed}) ===")
            print(f"==============================")

            cv_results = cross_validate_models(
                X,
                y,
                models,
                MODEL_REGISTRY,
                hyperparameter_tuning=hyperparameter_tuning,
                cv_folds=cv_folds,
                shuffle=cv_shuffle,
                random_state=seed,
            )

            # Print per-model mean ± std across folds
            print_cv_results(cv_results)

            # p-values for RF vs XGB using paired tests across folds
            if rf_name is not None and xgb_name is not None:
                print("\n--- RF vs XGB significance (paired across folds) ---")
                compare_models_significance(cv_results, rf_name, xgb_name, metric="mse")
                compare_models_significance(cv_results, rf_name, xgb_name, metric="mae")

    else:
        if need_ood:
            X_train, y_train, X_valid, y_valid, feature_columns = load_train_test_features_and_target(
                train_dataset_path,
                test_dataset_path,
            )
        else:
            X, y, feature_columns = load_features_and_target(dataset_path)
            X_train, X_valid, y_train, y_valid = split_dataset(X, y, train_size=0.8)

        # 2) Train models (optionally tuned)
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


        # 3) Report validation metrics
        print_holdout_results(y_valid, preds)



    # 4) Data visualization
    pt, mm = load_elemental_data(pt_path, mm_path)
    if data_visualization and not need_ood:
        X_raw = load_raw_data(dataset_path)
        plot_ms_distribution_by_tm(X_raw, save_path=plots_save_dir + f"{prefix}_ms_distribution_by_tm.png")
        plot_violin_ms_by_tm(X_raw, save_path=plots_save_dir + f"{prefix}_violin_ms_by_tm.png")
        summarize_compound_radix(X_raw, pt)

    # 5) Model interpretability and ablation analyses
    if ablation_study and not need_cross_validation:

        # Permutation feature importance evaluated on the validation set (Random Forest)
        if "rf" in trained_models:
            plot_permutation_importance(trained_models["rf"], X_valid, y_valid, title=f"RF Permutation Importance ({prefix})", 
                                    save_path=plots_save_dir + f"{prefix}_perm_importance_rf.png")

        # SHAP summary plot for global feature attribution (Random Forest)
        if "rf" in trained_models:
            plot_shap_summary(trained_models["rf"], X_train, X_valid, save_path=plots_save_dir + f"{prefix}_shap_summary_rf.png")

        # Comparative case studies across models (RF, XGBoost, and Ridge)
        if "rf" in trained_models and "xgb" in trained_models and "ridge" in trained_models:
            plot_case_studies(feature_columns, trained_models["rf"], trained_models["xgb"], trained_models["ridge"], pt, mm, 
                              save_path=plots_save_dir + f"{prefix}_case_studies.png")


if __name__ == "__main__":
    main()
