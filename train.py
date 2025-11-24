# train.py
"""
Training and tuning helpers for all models in models.py:
- Train linear, tree/boosting, kernel, and neural regressors
# - Optional GridSearchCV hyperparameter optimization
"""

from typing import Dict

from sklearn.model_selection import GridSearchCV
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

def tune_ridge_hyperparams(X_train, y_train) -> Dict:
    """Run GridSearchCV to search optimized Ridge hyperparameters."""
    param_grid = {
        "alpha": [0.01, 0.1, 1.0, 10.0, 50.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        "max_iter": [1000, 5000, 10000],
    }

    ridge_model = build_ridge_model()
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

def tune_lasso_hyperparams(X_train, y_train) -> Dict:
    """Run GridSearchCV to search optimized Lasso hyperparameters."""
    param_grid = {
        "alpha": [0.0005, 0.001, 0.01, 0.1, 1.0],
        "selection": ["cyclic", "random"],
        "max_iter": [1000, 5000, 10000],
    }

    lasso_model = build_lasso_model()
    grid_search = GridSearchCV(
        estimator=lasso_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Lasso Best Parameters:", best_params)
    print("Lasso Best Score (neg_mean_squared_error):", best_score)
    return best_params


def tune_elasticnet_hyperparams(X_train, y_train) -> Dict:
    """Run GridSearchCV to search optimized ElasticNet hyperparameters."""
    param_grid = {
        "alpha": [0.0005, 0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.2, 0.5, 0.8],
        "max_iter": [1000, 5000, 10000],
    }

    enet_model = build_elasticnet_model()
    grid_search = GridSearchCV(
        estimator=enet_model,
        param_grid=param_grid,
        cv=5,
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

def tune_rf_hyperparams(X_train, y_train) -> Dict:
    """Run GridSearchCV to search optimized Random Forest hyperparameters."""
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

def tune_xgb_hyperparams(X_train, y_train) -> Dict:
    """Run GridSearchCV to search optimized XGBoost hyperparameters."""
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

# 3) Hyperparameter tuning for kernel and neural network models

def tune_svr_hyperparams(X_train, y_train) -> Dict:
    """Run GridSearchCV to search optimized Support Vector Regressor hyperparameters."""
    param_grid = {
        "C": [0.5, 1.0, 5.0, 10.0],
        "epsilon": [0.01, 0.05, 0.1],
        "kernel": ["rbf", "poly", "sigmoid"],
    }

    svr_model = build_svr_model()
    grid_search = GridSearchCV(
        estimator=svr_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("SVR Best Parameters:", best_params)
    print("SVR Best Score (neg_mean_squared_error):", best_score)
    return best_params


def tune_mlp_hyperparams(X_train, y_train) -> Dict:
    """Run GridSearchCV to search optimized Multi-Layer Perceptron hyperparameters."""
    param_grid = {
        "hidden_layer_sizes": [(100,), (128, 64), (256, 128)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 1e-2],
        "early_stopping": [True, False],
    }

    mlp_model = build_mlp_model()
    grid_search = GridSearchCV(
        estimator=mlp_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("MLP Best Parameters:", best_params)
    print("MLP Best Score (neg_mean_squared_error):", best_score)
    return best_params


# 4) Training linear models

def train_linear_regression(X_train, y_train):
    """Train Linear Regression (no hyperparameters)."""
    model = build_linear_regression_model()
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train, y_train, params: Dict = None):
    """Train Ridge regression with searched optimized parameters or manually provided parameters."""
    if params is not None:
        ridge_model = build_ridge_model(**params)
    else:
        ridge_model = build_ridge_model(alpha=1.0, solver="lsqr")
    ridge_model.fit(X_train, y_train)
    return ridge_model


def train_lasso(X_train, y_train, params: Dict = None):
    """Train Lasso regression with searched optimized parameters or manually provided parameters."""
    if params is not None:
        lasso_model = build_lasso_model(**params)
    else:
        lasso_model = build_lasso_model(alpha=1.0, selection="cyclic")
    lasso_model.fit(X_train, y_train)
    return lasso_model


def train_elasticnet(X_train, y_train, params: Dict = None):
    """Train ElasticNet regression with searched optimized parameters or manually provided parameters."""
    if params is not None:
        enet_model = build_elasticnet_model(**params)
    else:
        enet_model = build_elasticnet_model(alpha=0.1, l1_ratio=0.5)
    enet_model.fit(X_train, y_train)
    return enet_model


# 5) Training tree/boosting models

def train_rf(X_train, y_train, params: Dict = None):
    """Train a random forest with searched optimized parameters or manually provided parameters."""
    if params is not None:
        rf_model = build_rf_model(**params)     
    else:
        rf_model = build_rf_model(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=1,
            min_samples_split=2,
        )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgb(X_train, y_train, params: Dict = None):
    """Train XGBoost with searched optimized parameters or manually provided parameters."""
    if params is not None:
        xgb_model = build_xgb_model(**params)
    else:
        xgb_model = build_xgb_model(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            subsample=0.6,
            colsample_bytree=0.8,
        )
    xgb_model.fit(X_train, y_train)
    return xgb_model

# 6) Training kernel and neural network models

def train_svr(X_train, y_train, params: Dict = None):
    """Train Support Vector Regression with searched optimized parameters or manually provided parameters."""
    if params is not None:
        svr_model = build_svr_model(**params)
    else:
        svr_model = build_svr_model(C=1.0, epsilon=0.1, kernel="rbf")
    svr_model.fit(X_train, y_train)
    return svr_model


def train_mlp(X_train, y_train, params: Dict = None):
    """Train MLP regression with searched optimized parameters or manually provided parameters."""
    if params is not None:
        mlp_model = build_mlp_model(**params)
    else:
        mlp_model = build_mlp_model(
            hidden_layer_sizes=(100,),
            activation="relu",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=False,
        )
    mlp_model.fit(X_train, y_train)
    return mlp_model