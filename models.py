"""
Build various machine learning regression models for predicting material properties.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


# 1) Linear Models
def build_linear_regression_model() -> LinearRegression:
    """Construct a Linear Regression model."""
    return LinearRegression()

def build_ridge_model(alpha: float = 1.0, solver: str = "auto", max_iter: int = 1000) -> Ridge:
    """Construct a Ridge Regression model."""
    return Ridge(alpha=alpha, solver=solver, max_iter=max_iter)

def build_lasso_model(alpha: float = 1.0, selection: str = "cyclic", max_iter: int = 1000, tol: float = 1e-4) -> Lasso:
    """Construct a Lasso Regression model."""
    return Lasso(alpha=alpha, selection=selection, max_iter=max_iter, tol=tol)

def build_elasticnet_model(alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000, tol: float = 1e-4) -> ElasticNet:
    """Construct an ElasticNet Regression model."""
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol)


# 2) Tree-based Models
def build_rf_model(
    n_estimators: int = 200,
    max_depth: int | None = 15,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | int | float | None = "auto",
    random_state: int = 0,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Construct a Random Forest Regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )

def build_xgb_model(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 7,
    min_child_weight: int = 1,
    gamma: float = 0.0,
    subsample: float = 0.6,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 0,
    n_jobs: int = -1,
    objective: str = "reg:squarederror",
) -> xgb.XGBRegressor:
    """Construct an XGB Regressor."""
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        objective=objective,
    )

def build_lightgbm_model(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = -1,
    num_leaves: int = 31,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_samples: int = 20,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    random_state: int = 0,
    n_jobs: int = -1,
    objective: str = "regression",
    boosting_type: str = "gbdt",
) -> lgb.LGBMRegressor:
    """Construct a LightGBM Regressor."""
    return lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        objective=objective,
        boosting_type=boosting_type,
    )

def build_catboost_model(
    iterations: int = 200,
    learning_rate: float = 0.05,
    depth: int = 7,
    l2_leaf_reg: float = 3.0,
    subsample: float = 0.8,
    colsample_bylevel: float = 0.8,
    random_seed: int = 0,
    loss_function: str = "RMSE",
) -> CatBoostRegressor:
    """Construct a CatBoost Regressor."""
    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        subsample=subsample,
        colsample_bylevel=colsample_bylevel,
        random_seed=random_seed,
        loss_function=loss_function,
        verbose=0,
    )

# 3) Kernel-based Models
def build_svr_model(
    C: float = 1.0,
    epsilon: float = 0.1,
    kernel: str = "rbf",
    gamma: str | float = "scale",
) -> SVR:
    """Construct an Support Vector Regression model."""
    return SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)


# 4) Neural Networks
def build_mlp_model(
    hidden_layer_sizes: tuple = (100,),
    activation: str = "relu",
    solver: str = "adam",
    alpha: float = 0.0001,
    learning_rate_init: float = 0.001,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    random_state: int = 0,
) -> MLPRegressor:
    """Construct a Multi-layer Perceptron model."""
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state,
    )






