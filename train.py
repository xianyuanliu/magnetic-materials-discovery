# train.py
"""
Training and tuning:
- Train RF / XGB / Ridge on a single dataset (X_train, y_train)
- Optionally run GridSearchCV for hyperparameter optimisation
"""

from typing import Dict, Tuple

from sklearn.model_selection import GridSearchCV
from models import build_rf_model, build_xgb_model, build_ridge_model


# ====== Random Forest ======

def tune_rf_hyperparams(X_train, y_train) -> Tuple[object, Dict]:
    """Run GridSearchCV to find strong RF hyperparameters."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rf_model = build_rf_model()
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("RF Best Parameters:", best_params)
    print("RF Best Score (neg_mean_squared_error):", best_score)
    return best_params


def train_rf(X_train, y_train, params: Dict = None):
    """Train a random forest; defaults to the tuned parameters used in this project."""
    if params is not None:
        rf_model = build_rf_model(random_state=0, **params)     
    else:
        rf_model = build_rf_model(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=0,
        )
    rf_model.fit(X_train, y_train)
    return rf_model


# ====== XGBoost ======

def tune_xgb_hyperparams(X_train, y_train):
    """Run GridSearchCV to search XGBoost hyperparameters."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    xgb_model = build_xgb_model()
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("XGB Best Parameters:", best_params)
    print("XGB Best Score (neg_mean_squared_error):", best_score)
    return best_params


def train_xgb(X_train, y_train, params: Dict = None):
    """Train XGBoost with project defaults or provided parameters."""
    if params is not None:
        xgb_model = build_xgb_model(random_state=0, **params)
    else:
        xgb_model = build_xgb_model(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            subsample=0.6,
            colsample_bytree=0.8,
            random_state=0,
        )
    xgb_model.fit(X_train, y_train)
    return xgb_model


# ====== Ridge ======

def tune_ridge_hyperparams(X_train, y_train):
    """Run GridSearchCV to search Ridge hyperparameters."""
    param_grid = {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    }

    ridge_model = build_ridge_model(alpha=1.0, solver="auto")
    grid_search = GridSearchCV(
        estimator=ridge_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Ridge Best Parameters:", best_params)
    print("Ridge Best Score (neg_mean_squared_error):", best_score)
    return best_params


def train_ridge(X_train, y_train, params: Dict = None):
    """Train Ridge regression with project defaults or provided parameters."""
    if params is not None:
        ridge_model = build_ridge_model(**params)
    else:
        ridge_model = build_ridge_model(alpha=1.0, solver="lsqr")
    ridge_model.fit(X_train, y_train)
    return ridge_model
