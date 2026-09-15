"""Regression models.

Each model is defined in a separate block covering model construction, hyperparameter tuning, and training.
``MODEL_REGISTRY`` at the bottom of the module maps each model to the corresponding configuration key.
"""

from typing import Dict, List, Optional, Union

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost

from utils.registry import ModelSpec


def _scaled(estimator) -> Pipeline:
    """Wrap `estimator` in a scikit-learn `Pipeline` with a preceding `StandardScaler` step."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def _prefix_hyperparams(hyperparams: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
    """Rewrite bare hyperparameter names for a scaled `Pipeline` ("alpha" -> "model__alpha").

    A list of dicts is how GridSearchCV expresses a union of sub-grids, so it is prefixed element-wise.
    """
    if isinstance(hyperparams, list):
        return [_prefix_hyperparams(sub) for sub in hyperparams]
    return {f"model__{key}": value for key, value in hyperparams.items()}


def _strip_hyperparams(hyperparams: Dict) -> Dict:
    """Inverse of _prefix_hyperparams, so tuned hyperparams stay usable as build_* kwargs."""
    return {key.split("__", 1)[-1]: value for key, value in hyperparams.items()}


def _grid_search_best_hyperparams(
    model,
    param_grid: Union[Dict, List[Dict]],
    X_train,
    y_train,
    cv_folds: int,
    label: str,
    scale: bool = False,
) -> Dict:
    """Run GridSearchCV for a given model instance, print and return the best hyperparams.

    Returns bare (un-prefixed) parameter names even when `scale` is set, so the
    result stays a valid kwargs dict for the matching build_* function.
    """
    grid_search = GridSearchCV(
        estimator=_scaled(model) if scale else model,
        param_grid=_prefix_hyperparams(param_grid) if scale else param_grid,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_hyperparams = _strip_hyperparams(grid_search.best_params_) if scale else grid_search.best_params_
    print(f"{label} Best Parameters:", best_hyperparams)
    print(f"{label} Best Score (neg_mean_squared_error):", grid_search.best_score_)
    return best_hyperparams


def _randomized_search_best_hyperparams(
    model,
    param_dist: Dict,
    X_train,
    y_train,
    cv_folds: int,
    random_state: int,
    label: str,
    n_iter: int = 20,
    scale: bool = False,
) -> Dict:
    """Run RandomizedSearchCV for a given model instance, print and return the best hyperparams.

    Returns bare (un-prefixed) parameter names even when `scale` is set; see _grid_search_best_hyperparams.
    """
    random_search = RandomizedSearchCV(
        estimator=_scaled(model) if scale else model,
        param_distributions=_prefix_hyperparams(param_dist) if scale else param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    random_search.fit(X_train, y_train)
    best_hyperparams = _strip_hyperparams(random_search.best_params_) if scale else random_search.best_params_
    print(f"{label} Best Parameters:", best_hyperparams)
    print(f"{label} Best Score (neg_mean_squared_error):", random_search.best_score_)
    return best_hyperparams


# --- Linear Regression ---

def build_linear_regression() -> LinearRegression:
    """Construct a Linear Regression model."""
    return LinearRegression()


def train_linear_regression(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Train Linear Regression."""
    model = _scaled(build_linear_regression())
    model.fit(X_train, y_train)
    return model


# --- Ridge ---

def build_ridge(alpha: float = 1.0) -> Ridge:
    """Construct a Ridge Regression model."""
    return Ridge(alpha=alpha, max_iter=10000)


def tune_ridge_hyperparams(
    X_train, y_train, cv_folds: int = 3, random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Run GridSearchCV to search optimized Ridge hyperparameters."""
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    }
    return _grid_search_best_hyperparams(build_ridge(), param_grid, X_train, y_train, cv_folds, "Ridge", scale=True)


def train_ridge(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Train Ridge."""
    if hyperparams is not None:
        model = build_ridge(**hyperparams)
    else:
        model = build_ridge(alpha=1.0)
    model = _scaled(model)
    model.fit(X_train, y_train)
    return model


# --- Lasso ---

def build_lasso(alpha: float = 0.01) -> Lasso:
    """Construct a Lasso Regression model."""
    return Lasso(alpha=alpha, max_iter=10000)


def tune_lasso_hyperparams(
    X_train, y_train, cv_folds: int = 3, random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Run GridSearchCV to search optimized Lasso hyperparameters."""
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    }
    return _grid_search_best_hyperparams(build_lasso(), param_grid, X_train, y_train, cv_folds, "Lasso", scale=True)


def train_lasso(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Train Lasso."""
    if hyperparams is not None:
        model = build_lasso(**hyperparams)
    else:
        model = build_lasso(alpha=0.001)
    model = _scaled(model)
    model.fit(X_train, y_train)
    return model


# --- ElasticNet ---

def build_elasticnet(alpha: float = 0.01, l1_ratio: float = 0.5) -> ElasticNet:
    """Construct an ElasticNet Regression model."""
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)


def tune_elasticnet_hyperparams(
    X_train, y_train, cv_folds: int = 3, random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Run GridSearchCV to search optimized ElasticNet hyperparameters."""
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    return _grid_search_best_hyperparams(
        build_elasticnet(), param_grid, X_train, y_train, cv_folds, "ElasticNet", scale=True
    )


def train_elasticnet(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Train ElasticNet."""
    if hyperparams is not None:
        model = build_elasticnet(**hyperparams)
    else:
        model = build_elasticnet(alpha=0.001, l1_ratio=0.1)
    model = _scaled(model)
    model.fit(X_train, y_train)
    return model


# --- Random Forest ---

def build_random_forest(
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


def tune_random_forest_hyperparams(
    X_train, y_train, cv_folds: int = 3, random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Run GridSearchCV to search optimized Random Forest hyperparameters.

    max_features is a fraction rather than "sqrt"/"log2": on the 9 engineered features both of those resolve to 3, which
    leaves 1.0 (consider every feature) unreachable.
    """
    param_grid = {
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.5, 0.75, 1.0],
    }
    return _grid_search_best_hyperparams(
        build_random_forest(random_state=random_state), param_grid, X_train, y_train, cv_folds, "RF"
    )


def train_random_forest(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Train a random forest."""
    if hyperparams is not None:
        model = build_random_forest(**hyperparams, random_state=random_state)
    else:
        model = build_random_forest(
            max_depth=15,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=random_state,
        )
    # Not wrapped in _scaled: scaling buys trees nothing and would cost SHAP its fast TreeExplainer.
    model.fit(X_train, y_train)
    return model


# --- XGBoost ---

def build_xgboost(
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


def tune_xgboost_hyperparams(
    X_train, y_train, cv_folds: int = 3, random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Run GridSearchCV to search optimized XGBoost hyperparameters.

    learning_rate is tied to n_estimators rather than crossed with it: each sub-grid holds their product roughly
    constant, so no candidate is simply an undertrained version of another. colsample_bytree is left out — with 9
    features there is little to subsample — while subsample stays, since subsampling rows is a real regularizer at
    n=460.
    """
    param_grid = [
        {
            "learning_rate": [learning_rate],
            "n_estimators": [n_estimators],
            "max_depth": [3, 5, 7],
            "reg_lambda": [1.0, 10.0],
            "subsample": [0.6, 0.8, 1.0],
        }
        for learning_rate, n_estimators in [(0.01, 1500), (0.05, 600), (0.1, 300)]
    ]
    return _grid_search_best_hyperparams(
        build_xgboost(random_state=random_state), param_grid, X_train, y_train, cv_folds, "XGB"
    )


def train_xgboost(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Train XGBoost."""
    if hyperparams is not None:
        model = build_xgboost(**hyperparams, random_state=random_state)
    else:
        model = build_xgboost(
            learning_rate=0.01,
            max_depth=7,
            min_child_weight=7,
            subsample=0.6,
            colsample_bytree=0.6,
            random_state=random_state,
        )
    # Not wrapped in _scaled, for the same reason as the random forest.
    model.fit(X_train, y_train)
    return model


# --- Support Vector Regression ---

def build_svr(
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


def tune_svr_hyperparams(
    X_train, y_train, cv_folds: int = 3, random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Run GridSearchCV to search optimized Support Vector Regressor hyperparameters.

    Restricted to the rbf kernel: scaled (see _scaled), rbf beats linear clearly, and gamma is meaningless for a linear
    kernel anyway.
    """
    param_grid = {
        "C": [1.0, 10.0, 100.0, 1000.0],
        "gamma": ["scale", 0.01, 0.1, 1.0],
        "epsilon": [0.01, 0.05, 0.1],
    }
    return _grid_search_best_hyperparams(build_svr(), param_grid, X_train, y_train, cv_folds, "SVR", scale=True)


def train_svr(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Train Support Vector Regression."""
    if hyperparams is not None:
        model = build_svr(**hyperparams)
    else:
        model = build_svr(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
    model = _scaled(model)
    model.fit(X_train, y_train)
    return model


# --- Multi-layer Perceptron ---

def build_mlp(
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


def tune_mlp_hyperparams(
    X_train, y_train, cv_folds: int = 3, random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Run RandomizedSearchCV to search optimized Multi-Layer Perceptron hyperparameters."""
    param_dist = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (256, 128), (128, 64, 32)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 1e-3, 5e-3, 1e-2],
    }
    return _randomized_search_best_hyperparams(
        build_mlp(random_state=random_state), param_dist, X_train, y_train, cv_folds,
        random_state, "MLP", n_iter=n_iter, scale=True,
    )


def train_mlp(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Train an MLP."""
    if hyperparams is not None:
        model = build_mlp(**hyperparams, random_state=random_state)
    else:
        model = build_mlp(
            hidden_layer_sizes=(256, 128),
            activation="tanh",
            alpha=1e-5,
            learning_rate_init=1e-3,
            random_state=random_state,
        )
    model = _scaled(model)
    model.fit(X_train, y_train)
    return model


# --- Model Registry ---

MODEL_REGISTRY = {spec.key: spec for spec in [
    ModelSpec("linear", "Linear Regression", train_linear_regression),

    ModelSpec("ridge", "Ridge", train_ridge, tune_ridge_hyperparams),
    ModelSpec("lasso", "Lasso", train_lasso, tune_lasso_hyperparams),
    ModelSpec("elasticnet", "ElasticNet", train_elasticnet, tune_elasticnet_hyperparams),

    ModelSpec("rf", "Random Forest", train_random_forest, tune_random_forest_hyperparams),
    ModelSpec("xgb", "XGBoost", train_xgboost, tune_xgboost_hyperparams),

    ModelSpec("svr", "SVR", train_svr, tune_svr_hyperparams),
    ModelSpec("mlp", "MLP", train_mlp, tune_mlp_hyperparams),
]}
