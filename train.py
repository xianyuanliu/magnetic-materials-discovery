"""
Training and tuning helpers for all models in models.py:
- Train linear, tree/boosting, kernel, and neural regressors
- GridSearchCV for small search spaces (Ridge, Lasso, ElasticNet)
- RandomizedSearchCV (n_iter=50) for large spaces (RF, XGBoost, SVR, MLP)

All tune_*/train_* functions accept a random_state so a single seed can be
threaded through model construction and hyperparameter search from the caller.
"""

import types
from typing import Dict

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from models import (
    build_linear_regression_model,
    build_ridge_model,
    build_lasso_model,
    build_elasticnet_model,
    build_rf_model,
    build_xgb_model,
    build_svr_model,
    build_mlp_model,
)


# 1) Hyperparameter tuning for linear models

def tune_ridge_hyperparams(X_train, y_train, cv_folds: int = 5, random_state: int = 0) -> Dict:
    """Run GridSearchCV to search optimized Ridge hyperparameters.

    random_state is accepted (but unused) so every MODEL_REGISTRY["tune"]
    callable shares the same call signature; Ridge/GridSearchCV has no
    stochastic element to seed.
    """
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    }

    ridge_model = build_ridge_model()
    grid_search = GridSearchCV(
        estimator=ridge_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Ridge Best Parameters:", best_params)
    print("Ridge Best Score (neg_mean_squared_error):", best_score)
    return best_params

def tune_lasso_hyperparams(X_train, y_train, cv_folds: int = 5, random_state: int = 0) -> Dict:
    """Run GridSearchCV to search optimized Lasso hyperparameters.

    random_state is accepted (but unused) for call-signature uniformity;
    see tune_ridge_hyperparams.
    """
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    }

    lasso_model = build_lasso_model()
    grid_search = GridSearchCV(
        estimator=lasso_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Lasso Best Parameters:", best_params)
    print("Lasso Best Score (neg_mean_squared_error):", best_score)
    return best_params


def tune_elasticnet_hyperparams(X_train, y_train, cv_folds: int = 5, random_state: int = 0) -> Dict:
    """Run GridSearchCV to search optimized ElasticNet hyperparameters.

    random_state is accepted (but unused) for call-signature uniformity;
    see tune_ridge_hyperparams.
    """
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    enet_model = build_elasticnet_model()
    grid_search = GridSearchCV(
        estimator=enet_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("ElasticNet Best Parameters:", best_params)
    print("ElasticNet Best Score (neg_mean_squared_error):", best_score)
    return best_params

# 2) Hyperparameter tuning for tree/boosting models

def tune_rf_hyperparams(X_train, y_train, cv_folds: int = 5, random_state: int = 0) -> Dict:
    """Run RandomizedSearchCV to search optimized Random Forest hyperparameters."""
    param_dist = {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    }

    rf_model = build_rf_model(random_state=random_state)
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("RF Best Parameters:", best_params)
    print("RF Best Score (neg_mean_squared_error):", best_score)
    return best_params

def tune_xgb_hyperparams(X_train, y_train, cv_folds: int = 5, random_state: int = 0) -> Dict:
    """Run RandomizedSearchCV to search optimized XGBoost hyperparameters."""
    param_dist = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    xgb_model = build_xgb_model(random_state=random_state)
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("XGB Best Parameters:", best_params)
    print("XGB Best Score (neg_mean_squared_error):", best_score)
    return best_params

# 3) Hyperparameter tuning for kernel and neural network models

def tune_svr_hyperparams(X_train, y_train, cv_folds: int = 5, random_state: int = 0) -> Dict:
    """Run RandomizedSearchCV to search optimized Support Vector Regressor hyperparameters.

    SVR itself has no random_state (deterministic solver); random_state here only
    seeds which hyperparameter combinations RandomizedSearchCV samples.
    """
    param_dist = {
        "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "epsilon": [0.01, 0.05, 0.1, 0.5],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
    }

    svr_model = build_svr_model()
    random_search = RandomizedSearchCV(
        estimator=svr_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    print("SVR Best Parameters:", best_params)
    print("SVR Best Score (neg_mean_squared_error):", best_score)
    return best_params


def tune_mlp_hyperparams(X_train, y_train, cv_folds: int = 5, random_state: int = 0) -> Dict:
    """Run RandomizedSearchCV to search optimized Multi-Layer Perceptron hyperparameters."""
    param_dist = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (256, 128), (128, 64, 32)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 1e-3, 5e-3, 1e-2],
    }

    mlp_model = build_mlp_model(random_state=random_state)
    random_search = RandomizedSearchCV(
        estimator=mlp_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    print("MLP Best Parameters:", best_params)
    print("MLP Best Score (neg_mean_squared_error):", best_score)
    return best_params


# 4) Training linear models

def train_linear_regression(X_train, y_train, params: Dict = None, random_state: int = 0):
    """Train Linear Regression (no hyperparameters, no randomness to seed)."""
    del random_state  # unused; accepted for a uniform MODEL_REGISTRY["train"] signature
    model = build_linear_regression_model()
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train, y_train, params: Dict = None, random_state: int = 0):
    """Train Ridge regression with searched optimized parameters or manually provided parameters."""
    del random_state  # unused; Ridge's default solver is deterministic
    if params is not None:
        ridge_model = build_ridge_model(**params)
    else:
        ridge_model = build_ridge_model(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    return ridge_model


def train_lasso(X_train, y_train, params: Dict = None, random_state: int = 0):
    """Train Lasso regression with searched optimized parameters or manually provided parameters."""
    del random_state  # unused; default selection="cyclic" is deterministic
    if params is not None:
        lasso_model = build_lasso_model(**params)
    else:
        lasso_model = build_lasso_model(alpha=0.001)
    lasso_model.fit(X_train, y_train)
    return lasso_model


def train_elasticnet(X_train, y_train, params: Dict = None, random_state: int = 0):
    """Train ElasticNet regression with searched optimized parameters or manually provided parameters."""
    del random_state  # unused; default selection="cyclic" is deterministic
    if params is not None:
        enet_model = build_elasticnet_model(**params)
    else:
        enet_model = build_elasticnet_model(alpha=0.001, l1_ratio=0.1)
    enet_model.fit(X_train, y_train)
    return enet_model


# 5) Training tree/boosting models

def train_rf(X_train, y_train, params: Dict = None, random_state: int = 0):
    """Train a random forest with searched optimized parameters or manually provided parameters."""
    if params is not None:
        rf_model = build_rf_model(**params, random_state=random_state)
    else:
        rf_model = build_rf_model(
            max_depth=15,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
        )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgb(X_train, y_train, params: Dict = None, random_state: int = 0):
    """Train XGBoost with searched optimized parameters or manually provided parameters."""
    if params is not None:
        xgb_model = build_xgb_model(**params, random_state=random_state)
    else:
        xgb_model = build_xgb_model(
            learning_rate=0.01,
            max_depth=7,
            min_child_weight=7,
            subsample=0.6,
            colsample_bytree=0.6,
            random_state=random_state,
        )
    xgb_model.fit(X_train, y_train)
    return xgb_model

# 6) Training kernel and neural network models

def train_svr(X_train, y_train, params: Dict = None, random_state: int = 0):
    """Train Support Vector Regression with searched optimized parameters or manually provided parameters."""
    del random_state  # unused; SVR has no random_state (deterministic solver)
    if params is not None:
        svr_model = build_svr_model(**params)
    else:
        svr_model = build_svr_model(C=0.1, epsilon=0.1, kernel="linear", gamma="scale")
    svr_model.fit(X_train, y_train)
    return svr_model


def train_mlp(X_train, y_train, params: Dict = None, random_state: int = 0):
    """Train MLP regression with searched optimized parameters or manually provided parameters."""
    if params is not None:
        mlp_model = build_mlp_model(**params, random_state=random_state)
    else:
        mlp_model = build_mlp_model(
            hidden_layer_sizes=(256, 128),
            activation="tanh",
            alpha=1e-5,
            learning_rate_init=1e-4,
            random_state=random_state,
        )
    mlp_model.fit(X_train, y_train)
    return mlp_model


MODEL_REGISTRY = types.MappingProxyType({
    "linear": {"name": "Linear Regression", "train": train_linear_regression, "tune": None},

    "ridge": {"name": "Ridge", "train": train_ridge, "tune": tune_ridge_hyperparams},
    "lasso": {"name": "Lasso", "train": train_lasso, "tune": tune_lasso_hyperparams},
    "elasticnet": {"name": "ElasticNet", "train": train_elasticnet, "tune": tune_elasticnet_hyperparams},

    "rf": {"name": "Random Forest", "train": train_rf, "tune": tune_rf_hyperparams},
    "xgb": {"name": "XGBoost", "train": train_xgb, "tune": tune_xgb_hyperparams},

    "svr": {"name": "SVR", "train": train_svr, "tune": tune_svr_hyperparams},
    "mlp": {"name": "MLP", "train": train_mlp, "tune": tune_mlp_hyperparams},
})