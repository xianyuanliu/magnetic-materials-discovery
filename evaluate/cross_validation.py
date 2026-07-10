"""K-fold cross-validation, holdout reporting, and paired significance testing."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import KFold

from evaluate.metrics import compute_metrics


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_keys: List[str],
    model_registry: Dict,
    hyperparameter_tuning: bool = False,
    best_params: Optional[Dict] = None,
    cv_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
    model_random_state: int = 0,
    report_rf_xgb: bool = True,
):
    """Run K-fold cross-validation for the requested models.

    random_state seeds the KFold split; model_random_state seeds model
    construction and hyperparameter search so the two sources of randomness
    can be controlled independently.
    """
    results = {}
    for key in model_keys:
        if key not in model_registry:
            raise ValueError(f"Unknown model key: {key}")
        name = model_registry[key]["name"]
        results[name] = {"mse": [], "mae": [], "mre": [], "r2": []}

    kf = KFold(
        n_splits=cv_folds,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )

    # Track RF vs XGB per-fold MSE (only if requested and both models exist)
    track_rf_xgb = report_rf_xgb and ("rf" in model_keys) and ("xgb" in model_keys)
    rf_fold_mse = []
    xgb_fold_mse = []

    for train_idx, valid_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        for key in model_keys:
            model_cfg = model_registry[key]
            params = None
            if best_params is not None and key in best_params:
                params = best_params[key]
            elif hyperparameter_tuning and model_cfg["tune"] is not None:
                params = model_cfg["tune"](
                    X_train, y_train, cv_folds=cv_folds, random_state=model_random_state
                )

            model = model_cfg["train"](X_train, y_train, params=params, random_state=model_random_state)
            y_pred = model.predict(X_valid)

            metrics = compute_metrics(y_valid, y_pred)

            name = model_cfg["name"]
            results[name]["mse"].append(metrics["mse"])
            results[name]["mae"].append(metrics["mae"])
            results[name]["mre"].append(metrics["mre"])
            results[name]["r2"].append(metrics["r2"])

            if track_rf_xgb:
                if key == "rf":
                    rf_fold_mse.append(metrics["mse"])
                elif key == "xgb":
                    xgb_fold_mse.append(metrics["mse"])

    # how many times RF outperforms XGB
    if track_rf_xgb and len(rf_fold_mse) == cv_folds and len(xgb_fold_mse) == cv_folds:
        wins_rf = sum(m_rf < m_xgb for m_rf, m_xgb in zip(rf_fold_mse, xgb_fold_mse))
        wins_xgb = sum(m_xgb < m_rf for m_rf, m_xgb in zip(rf_fold_mse, xgb_fold_mse))
        ties = cv_folds - wins_rf - wins_xgb

        print("\nFold-by-fold win count (metric=MSE): Random Forest vs XGBoost")
        print(f"RF wins:  {wins_rf}/{cv_folds}")
        print(f"XGB wins: {wins_xgb}/{cv_folds}")
        print(f"Ties:     {ties}/{cv_folds}")

    return results

# ====== Quantitative metrics ======

def print_holdout_results(y_true, predictions: Dict[str, np.ndarray]):
    """Print MSE, MAE, and R² for multiple regression models."""
    print("Regression Metrics:")
    for name, y_pred in predictions.items():
        metrics = compute_metrics(y_true, y_pred)
        print(f"\n{name}:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MRE: {metrics['mre']:.6f}")
        print(f"R2:  {metrics['r2']:.4f}")

def print_cv_results(results: Dict[str, Dict[str, List[float]]]):
    """Print mean ± std metrics for cross-validation results."""
    print("Cross-Validation Metrics (mean ± std):")
    for name, scores in results.items():
        mse_mean = np.mean(scores["mse"])
        mse_std = np.std(scores["mse"], ddof=1)
        mae_mean = np.mean(scores["mae"])
        mae_std = np.std(scores["mae"], ddof=1)
        mre_mean = np.mean(scores["mre"])
        mre_std = np.std(scores["mre"], ddof=1)
        r2_mean = np.mean(scores["r2"])
        r2_std = np.std(scores["r2"], ddof=1)

        print(f"\n{name}:")
        print(f"MSE: {mse_mean:.4f} ± {mse_std:.4f}")
        print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
        print(f"MRE: {mre_mean:.6f} ± {mre_std:.6f}")
        print(f"R2:  {r2_mean:.4f} ± {r2_std:.4f}")

def compare_models_significance(
    results: Dict[str, Dict[str, List[float]]],
    model_a: str,
    model_b: str,
    metric: str = "mse",
):
    """Paired t-test and Wilcoxon signed-rank test on per-fold CV scores.

    Returns:
        (t_stat, t_pvalue, wilcoxon_stat, wilcoxon_pvalue).
    """

    if model_a not in results or model_b not in results:
        raise ValueError(f"Model names not found in results: {model_a}, {model_b}")

    a = np.array(results[model_a][metric], dtype=float)
    b = np.array(results[model_b][metric], dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Fold count mismatch: {model_a} has {len(a)}, {model_b} has {len(b)}")

    diff = a - b  # positive means A worse than B for MSE/MAE (lower is better)

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(a, b, nan_policy="omit")

    # Wilcoxon signed-rank (requires non-zero diffs)
    nonzero = diff[diff != 0]
    if len(nonzero) < 1:
        w_stat, w_p = np.nan, np.nan
    else:
        # Two-sided by default
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox")

    print(f"\nSignificance tests (paired) on CV folds — metric={metric}")
    print(f"  Comparing: {model_a} vs {model_b}")
    print(f"  Paired t-test:     t={t_stat:.4f}, p={t_p:.6g}")
    print(f"  Wilcoxon signed-rank: W={w_stat}, p={w_p:.6g}")

    return float(t_stat), float(t_p), float(w_stat), float(w_p)
