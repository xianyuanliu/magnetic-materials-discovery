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

def build_lasso_model(alpha: float = 1.0, selection: str = "cyclic", max_iter: int = 1000) -> Lasso:
    """Construct a Lasso Regression model."""
    return Lasso(alpha=alpha, selection=selection, max_iter=max_iter)

def build_elasticnet_model(alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000) -> ElasticNet:
    """Construct an ElasticNet Regression model."""
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)

# 2) Tree-based Models
def build_rf_model(
    n_estimators: int = 200,
    max_depth: int | None = 15,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> RandomForestRegressor:
    """Construct a Random Forest Regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

def build_xgb_model(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 7,
    min_child_weight: int = 1,
    subsample: float = 0.6,
    colsample_bytree: float = 0.8,
) -> xgb.XGBRegressor:
    """Construct an XGB Regressor."""
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
    )

# 3) Kernel-based Models
def build_svr_model(
    C: float = 1.0,
    epsilon: float = 0.1,
    kernel: str = "rbf",
) -> SVR:
    """Construct an Support Vector Regression model."""
    return SVR(C=C, epsilon=epsilon, kernel=kernel)


# 4) Neural Networks
def build_mlp_model(
    hidden_layer_sizes: tuple = (256, 128),
    activation: str = "relu",
    alpha: float = 0.01,
    learning_rate_init: float = 0.001,
    max_iter: int = 1000,
    early_stopping: bool = False,
) -> MLPRegressor:
    """Construct a Multi-layer Perceptron model."""
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
    )
