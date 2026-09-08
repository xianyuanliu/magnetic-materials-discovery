"""Builders for the regression models used to predict material properties."""

from typing import Optional, Union

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import xgboost


# 1) Linear Models
def build_linear_regression_model() -> LinearRegression:
    """Construct a Linear Regression model."""
    return LinearRegression()


def build_ridge_model(alpha: float = 1.0) -> Ridge:
    """Construct a Ridge Regression model."""
    return Ridge(alpha=alpha, max_iter=10000)


def build_lasso_model(alpha: float = 0.01) -> Lasso:
    """Construct a Lasso Regression model."""
    return Lasso(alpha=alpha, max_iter=10000)


def build_elasticnet_model(alpha: float = 0.01, l1_ratio: float = 0.5) -> ElasticNet:
    """Construct an ElasticNet Regression model."""
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)


# 2) Tree-based Models
def build_rf_model(
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: Union[str, float] = "sqrt",
    random_state: int = 0,
) -> RandomForestRegressor:
    """Construct a Random Forest Regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
    )


def build_xgb_model(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0,
    reg_alpha: float = 0,
    reg_lambda: float = 1.0,
    random_state: int = 0,
) -> xgboost.XGBRegressor:
    """Construct an XGB Regressor."""
    return xgboost.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        verbosity=0,
    )


# 3) Kernel-based Models
def build_svr_model(
    C: float = 10.0,
    epsilon: float = 0.1,
    kernel: str = "rbf",
    gamma: str = "scale",
    degree: int = 3,
) -> SVR:
    """Construct an Support Vector Regression model.

    sklearn's SVR has no random_state parameter: epsilon-SVR is solved by a
    deterministic dual algorithm (libsvm), so there is no stochasticity to seed.
    """
    return SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma, degree=degree)


# 4) Neural Networks
def build_mlp_model(
    hidden_layer_sizes: tuple = (128, 64),
    activation: str = "relu",
    alpha: float = 1e-4,
    learning_rate: str = "constant",
    learning_rate_init: float = 1e-3,
    max_iter: int = 1000,
    early_stopping: bool = True,
    random_state: int = 0,
) -> MLPRegressor:
    """Construct a Multi-layer Perceptron model."""
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state,
    )
