"""Run the magnetism pipeline for Novamag or Materials Project data."""

import argparse
from pathlib import Path
from typing import List

import yaml

from loaddata.feature_csv_access import load_features_and_target, load_raw_data, split_dataset
from loaddata.raw_loaders import load_elemental_data

from pipeline.train import MODEL_REGISTRY, DEFAULT_TUNE_CV_FOLDS, DEFAULT_TUNE_N_ITER
from pipeline.ood_pipeline import run_ood_evaluation
from pipeline.uq_pipeline import run_uq_evaluation

from evaluate.cross_validation import (
    print_holdout_results,
    cross_validate_models,
    print_cv_results,
    compare_models_significance,
)
from evaluate.metrics import compute_metrics

from interpret.model_weights import plot_permutation_importance, plot_shap_summary
from interpret.case_studies import plot_case_studies
from interpret.visualize import plot_ms_distribution_by_tm, plot_violin_ms_by_tm, summarize_compound_radix


def parse_args():
    """Parse the --config CLI flag."""
    parser = argparse.ArgumentParser(description="Train ML models for material property prediction")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/novamag.yaml",
        help="Path to YAML configuration file"
    )
    return parser.parse_args()


def load_config(path: str):
    """Load a YAML run config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_rf_vs_xgb_significance(cv_results, rf_name: str, xgb_name: str):
    """Print paired t-test / Wilcoxon significance for RF vs XGB, on MSE and MAE."""
    print("\n--- RF vs XGB significance (paired across folds) ---")
    for metric in ("mse", "mae"):
        t_stat, t_p, w_stat, w_p = compare_models_significance(
            cv_results, rf_name, xgb_name, metric=metric
        )
        print(
            f"{metric.upper()}: t-test p={t_p:.6g}, Wilcoxon p={w_p:.6g} "
            f"(t_stat={t_stat:.4f}, w_stat={w_stat:.4f})"
        )


def run_cross_validation(
    *,
    dataset_path: str,
    models: List[str],
    cv_folds: int,
    cv_shuffle: bool,
    cv_seeds: List[int],
    hyperparameter_tuning: bool,
    model_random_state: int,
    tune_cv_folds: int,
    tune_n_iter: int,
) -> None:
    """Run K-fold CV once per seed in cv_seeds; print metrics and RF-vs-XGB significance."""
    if len(cv_seeds) < 1:
        raise ValueError("cv_seeds must contain at least one seed.")

    X, y, _ = load_features_and_target(dataset_path)

    if hyperparameter_tuning:
        # The search is nested inside every outer fold, so its cost multiplies
        # out fast; surface the budget before spending an hour on it.
        tunable = [k for k in models if MODEL_REGISTRY[k]["tune"] is not None]
        searches = len(cv_seeds) * cv_folds * len(tunable)
        print(
            f"\n[INFO] Nested hyperparameter search: {len(cv_seeds)} seed(s) x "
            f"{cv_folds} folds x {len(tunable)} tunable model(s) = {searches} searches, "
            f"each up to {tune_n_iter} candidates x {tune_cv_folds} inner folds "
            f"(~{searches * tune_n_iter * tune_cv_folds} model fits). "
            f"Lower tune_n_iter / tune_cv_folds in the config to shrink this."
        )

    def _name_for_key(model_key: str) -> str:
        return MODEL_REGISTRY[model_key]["name"]

    rf_name = _name_for_key("rf") if "rf" in models else None
    xgb_name = _name_for_key("xgb") if "xgb" in models else None

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
            best_params=None,
            cv_folds=cv_folds,
            shuffle=cv_shuffle,
            random_state=seed,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )

        print_cv_results(cv_results)

        if rf_name is not None and xgb_name is not None:
            print_rf_vs_xgb_significance(cv_results, rf_name, xgb_name)


def run_holdout(
    *,
    dataset_path: str,
    pt_path: str,
    mm_path: str,
    models: List[str],
    holdout_seeds: List[int],
    train_size: float,
    hyperparameter_tuning: bool,
    model_random_state: int,
    tune_cv_folds: int,
    tune_n_iter: int,
    ablation_study: bool,
    prefix: str,
    plots_save_dir: Path,
) -> None:
    """Repeat a train/validate split once per seed in holdout_seeds; report mean ± std.

    holdout_seeds seeds the train/valid split only, and is deliberately
    separate from model_random_state (which seeds model construction and the
    hyperparameter search). A single split is one draw from a 460-sample
    dataset, so its metrics move by more than the gaps between models; several
    seeds give a spread to compare against.

    Ablation plots are produced from the first seed's models only, so figure
    filenames stay stable across runs.
    """
    if len(holdout_seeds) < 1:
        raise ValueError("holdout_seeds must contain at least one seed.")
    for key in models:
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {key}")

    X, y, feature_columns = load_features_and_target(dataset_path)

    scores = {MODEL_REGISTRY[key]["name"]: {"mse": [], "mae": [], "mre": [], "r2": []} for key in models}
    first_split_models = {}

    for run_i, seed in enumerate(holdout_seeds, start=1):
        seed = int(seed)
        print(f"\n=== Holdout Run {run_i}/{len(holdout_seeds)} (split seed={seed}) ===")

        X_train, X_valid, y_train, y_valid = split_dataset(
            X, y, train_size=train_size, random_state=seed
        )

        best_params = {}
        if hyperparameter_tuning:
            print("--- Hyperparameter Tuning (on this split's training set) ---")
            for key in models:
                model_cfg = MODEL_REGISTRY[key]
                if model_cfg["tune"] is not None:
                    best_params[key] = model_cfg["tune"](
                        X_train,
                        y_train,
                        cv_folds=tune_cv_folds,
                        random_state=model_random_state,
                        n_iter=tune_n_iter,
                    )

        trained_models = {}
        preds = {}
        for key in models:
            model_cfg = MODEL_REGISTRY[key]
            model = model_cfg["train"](
                X_train, y_train, params=best_params.get(key), random_state=model_random_state
            )
            trained_models[key] = model
            preds[model_cfg["name"]] = model.predict(X_valid)

        print_holdout_results(y_valid, preds)

        for name, y_pred in preds.items():
            for metric, value in compute_metrics(y_valid, y_pred).items():
                scores[name][metric].append(value)

        if not first_split_models:
            first_split_models = trained_models

    if len(holdout_seeds) > 1:
        print(f"\n=== Holdout across {len(holdout_seeds)} splits ===")
        print_cv_results(scores, title="Holdout Metrics (mean ± std over split seeds):")

    if not ablation_study:
        return

    trained_models = first_split_models
    X_train, X_valid, y_train, y_valid = split_dataset(
        X, y, train_size=train_size, random_state=int(holdout_seeds[0])
    )

    pt, mm = load_elemental_data(pt_path, mm_path)

    # Permutation importance and SHAP summary for Random Forest
    if "rf" in trained_models:
        plot_permutation_importance(
            trained_models["rf"], X_valid, y_valid, title=f"RF Permutation Importance ({prefix})",
            save_path=plots_save_dir / f"{prefix}_perm_importance_rf.png",
        )
        plot_shap_summary(
            trained_models["rf"], X_train, X_valid, save_path=plots_save_dir / f"{prefix}_shap_summary_rf.png"
        )

    # Comparative case studies across models (RF, XGBoost, and Ridge)
    if "rf" in trained_models and "xgb" in trained_models and "ridge" in trained_models:
        plot_case_studies(
            feature_columns, trained_models["rf"], trained_models["xgb"], trained_models["ridge"], pt, mm,
            save_path=plots_save_dir / f"{prefix}_case_studies.png",
        )


def main():
    """Run one holdout/cross_validation/ood evaluation, per the --config file."""
    plots_save_dir = Path("./plots/")

    args = parse_args()
    cfg = load_config(args.config)
    plots_save_dir.mkdir(parents=True, exist_ok=True)

    pt_path = cfg["pt_path"]
    mm_path = cfg["mm_path"]
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

    # Seeds model construction and hyperparameter search everywhere (independent
    # of cv_random_state/cv_seeds/holdout_seeds, which seed data splitting).
    model_random_state = int(cfg.get("random_state", 0))

    # Holdout splits are seeded separately from the models, and repeated, so a
    # single lucky/unlucky 80/20 draw can't be mistaken for a model difference.
    holdout_seeds = cfg.get("holdout_seeds", [cv_random_state])
    holdout_train_size = float(cfg.get("holdout_train_size", 0.8))

    # Hyperparameter-search budget, kept separate from cv_folds: the search is
    # nested inside every outer fold/split, so cv_folds would multiply its cost.
    tune_cv_folds = int(cfg.get("tune_cv_folds", DEFAULT_TUNE_CV_FOLDS))
    tune_n_iter = int(cfg.get("tune_n_iter", DEFAULT_TUNE_N_ITER))

    if evaluation_mode not in {"holdout", "cross_validation", "ood", "uq"}:
        raise ValueError(
            "Invalid evaluation_mode. Choose 'holdout', 'cross_validation', 'ood', or 'uq'."
        )
    need_cross_validation = evaluation_mode == "cross_validation"
    need_ood = evaluation_mode == "ood"
    need_uq = evaluation_mode == "uq"
    if need_ood and (not train_dataset_path or not test_dataset_path):
        raise ValueError("OOD mode requires train_dataset_path and test_dataset_path in the config.")
    if not need_ood and not dataset_path:
        raise ValueError("dataset_path is required for holdout, cross_validation, or uq modes.")

    if dataset_name == "novamag":
        prefix = "novamag"
    elif dataset_name == "mp":
        prefix = "mp"
    else:
        raise ValueError("Invalid dataset name. Choose either 'Novamag' or 'Materials Project'.")

    if need_cross_validation:
        run_cross_validation(
            dataset_path=dataset_path,
            models=models,
            cv_folds=cv_folds,
            cv_shuffle=cv_shuffle,
            cv_seeds=cv_seeds,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )
    elif need_ood:
        run_ood_evaluation(
            cfg=cfg,
            train_dataset_path=train_dataset_path,
            test_dataset_path=test_dataset_path,
            pt_path=pt_path,
            models=models,
            model_registry=MODEL_REGISTRY,
            cv_folds=cv_folds,
            cv_shuffle=cv_shuffle,
            hyperparameter_tuning=hyperparameter_tuning,
            cv_random_state=cv_random_state,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )
    elif need_uq:
        run_uq_evaluation(
            cfg=cfg,
            dataset_path=dataset_path,
            pt_path=pt_path,
            model_registry=MODEL_REGISTRY,
            cv_folds=cv_folds,
            cv_shuffle=cv_shuffle,
            cv_random_state=cv_random_state,
            model_random_state=model_random_state,
        )
    else:
        run_holdout(
            dataset_path=dataset_path,
            pt_path=pt_path,
            mm_path=mm_path,
            models=models,
            holdout_seeds=holdout_seeds,
            train_size=holdout_train_size,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
            ablation_study=ablation_study,
            prefix=prefix,
            plots_save_dir=plots_save_dir,
        )

    # Data visualization (holdout and cross_validation modes only)
    if data_visualization and not (need_ood or need_uq):
        X_raw = load_raw_data(dataset_path)
        plot_ms_distribution_by_tm(X_raw, save_path=plots_save_dir / f"{prefix}_ms_distribution_by_tm.png")
        plot_violin_ms_by_tm(X_raw, title=f"{prefix.upper()} Violin Plot", save_path=plots_save_dir / f"{prefix}_violin_ms_by_tm.png")
        summarize_compound_radix(X_raw)


if __name__ == "__main__":
    main()
