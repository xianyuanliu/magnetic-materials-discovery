"""
Training and tuning helpers for all models in pipeline/models.py:
- Train linear, tree/boosting, kernel, and neural regressors
- GridSearchCV, exhaustive, for the trimmed grids (Ridge, Lasso, ElasticNet,
  RF, XGBoost, SVR)
- RandomizedSearchCV for the one large space left (MLP)

Combines models.py's bare model builders with hyperparameter search into
ready-to-fit units (MODEL_REGISTRY), keyed by a single random_state threaded through
model construction and hyperparameter search from the caller. Each entry is a
core.ModelSpec, so a model declares what it can do (e.g. whether it exposes an
ensemble spread the UQ pipeline can read) instead of being recognised by key
elsewhere in the codebase.

Scale-sensitive models (linear, kernel, neural) are wrapped in a
StandardScaler pipeline; see _scaled for why the tree ensembles are not.
Search cost is controlled by the caller via cv_folds and n_iter, because a
nested search re-runs for every outer fold (see evaluate/cross_validation.py).
"""

from typing import Dict, List, Optional, Union

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import DEFAULT_TUNE_CV_FOLDS, DEFAULT_TUNE_N_ITER, ModelSpec, build_registry
from pipeline.models import (
    build_linear_regression_model,
    build_ridge_model,
    build_lasso_model,
    build_elasticnet_model,
    build_rf_model,
    build_xgb_model,
    build_svr_model,
    build_mlp_model,
)

# One hyperparameter grid, or the union of several sub-grids that
# GridSearchCV expresses as a list (see tune_xgb_hyperparams).
ParamGrid = Union[Dict, List[Dict]]

# Name of the estimator step inside the StandardScaler pipeline.
_ESTIMATOR_STEP = "model"


def _scaled(estimator) -> Pipeline:
    """Wrap `estimator` in a StandardScaler pipeline.

    The engineered features span three orders of magnitude (meltingTw ~1.7e3
    vs electronegw ~1.8), which cripples every distance-, penalty-, or
    gradient-based model: unscaled, SVR(rbf) and MLP score R^2 ~0.1 on
    Novamag versus ~0.6-0.7 scaled, and an rbf kernel can never win the
    hyperparameter search.

    Tree ensembles (RF, XGBoost) are deliberately left bare: they are
    invariant to per-feature monotone rescaling, so scaling buys nothing, and
    wrapping them in a Pipeline would stop shap.Explainer from dispatching to
    the fast exact TreeExplainer in interpret/model_weights.py.

    Scaling lives inside the pipeline (not applied to the whole dataset up
    front) so the mean/std are fitted on training data only and never leak
    across a train/test boundary.
    """
    return Pipeline([("scaler", StandardScaler()), (_ESTIMATOR_STEP, estimator)])


def _prefix_params(params: ParamGrid) -> ParamGrid:
    """Rewrite bare param names for a scaled pipeline ("alpha" -> "model__alpha").

    Accepts a list of dicts too, which is how GridSearchCV expresses a union of
    sub-grids (see tune_xgb_hyperparams, which ties learning_rate to
    n_estimators instead of taking their full product).
    """
    if isinstance(params, list):
        return [_prefix_params(sub) for sub in params]
    return {f"{_ESTIMATOR_STEP}__{key}": value for key, value in params.items()}


def _strip_params(params: Dict) -> Dict:
    """Inverse of _prefix_params, so tuned params stay usable as build_*_model kwargs."""
    return {key.split("__", 1)[-1]: value for key, value in params.items()}


def _grid_search_best_params(
    model,
    param_grid: ParamGrid,
    X_train,
    y_train,
    cv_folds: int,
    label: str,
    scale: bool = False,
) -> Dict:
    """Run GridSearchCV for a given model instance, print and return the best params.

    Returns bare (un-prefixed) parameter names even when `scale` is set, so the
    result stays a valid kwargs dict for the matching models.py builder.
    """
    grid_search = GridSearchCV(
        estimator=_scaled(model) if scale else model,
        param_grid=_prefix_params(param_grid) if scale else param_grid,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_params = _strip_params(grid_search.best_params_) if scale else grid_search.best_params_
    print(f"{label} Best Parameters:", best_params)
    print(f"{label} Best Score (neg_mean_squared_error):", grid_search.best_score_)
    return best_params


def _randomized_search_best_params(
    model,
    param_dist: Dict,
    X_train,
    y_train,
    cv_folds: int,
    random_state: int,
    label: str,
    n_iter: int = DEFAULT_TUNE_N_ITER,
    scale: bool = False,
) -> Dict:
    """Run RandomizedSearchCV for a given model instance, print and return the best params.

    Returns bare (un-prefixed) parameter names even when `scale` is set; see
    _grid_search_best_params.
    """
    random_search = RandomizedSearchCV(
        estimator=_scaled(model) if scale else model,
        param_distributions=_prefix_params(param_dist) if scale else param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    random_search.fit(X_train, y_train)
    best_params = _strip_params(random_search.best_params_) if scale else random_search.best_params_
    print(f"{label} Best Parameters:", best_params)
    print(f"{label} Best Score (neg_mean_squared_error):", random_search.best_score_)
    return best_params


# 1) Hyperparameter tuning for linear models
def tune_ridge_hyperparams(
    X_train, y_train, cv_folds: int = DEFAULT_TUNE_CV_FOLDS, random_state: int = 0,
    n_iter: int = DEFAULT_TUNE_N_ITER,
) -> Dict:
    """Run GridSearchCV to search optimized Ridge hyperparameters.

    random_state and n_iter are accepted (but unused) so every
    MODEL_REGISTRY["tune"] callable shares the same call signature;
    Ridge/GridSearchCV has no stochastic element to seed and enumerates its
    whole grid rather than sampling from it.
    """
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    }
    return _grid_search_best_params(
        build_ridge_model(), param_grid, X_train, y_train, cv_folds, "Ridge", scale=True
    )


def tune_lasso_hyperparams(
    X_train, y_train, cv_folds: int = DEFAULT_TUNE_CV_FOLDS, random_state: int = 0,
    n_iter: int = DEFAULT_TUNE_N_ITER,
) -> Dict:
    """Run GridSearchCV to search optimized Lasso hyperparameters.

    random_state and n_iter are accepted (but unused) for call-signature
    uniformity; see tune_ridge_hyperparams.
    """
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    }
    return _grid_search_best_params(
        build_lasso_model(), param_grid, X_train, y_train, cv_folds, "Lasso", scale=True
    )


def tune_elasticnet_hyperparams(
    X_train, y_train, cv_folds: int = DEFAULT_TUNE_CV_FOLDS, random_state: int = 0,
    n_iter: int = DEFAULT_TUNE_N_ITER,
) -> Dict:
    """Run GridSearchCV to search optimized ElasticNet hyperparameters.

    random_state and n_iter are accepted (but unused) for call-signature
    uniformity; see tune_ridge_hyperparams.
    """
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    return _grid_search_best_params(
        build_elasticnet_model(), param_grid, X_train, y_train, cv_folds, "ElasticNet", scale=True
    )


# 2) Hyperparameter tuning for tree/boosting models
def tune_rf_hyperparams(
    X_train, y_train, cv_folds: int = DEFAULT_TUNE_CV_FOLDS, random_state: int = 0,
    n_iter: int = DEFAULT_TUNE_N_ITER,
) -> Dict:
    """Run GridSearchCV to search optimized Random Forest hyperparameters.

    27 combinations, searched exhaustively — cheaper than sampling 20 points
    out of the old 45 and no longer luck-dependent. n_iter is accepted (but
    unused) for call-signature uniformity.

    max_features is a fraction rather than "sqrt"/"log2": on the 9 engineered
    features both of those resolve to int(sqrt(9)) == int(log2(9)) == 3, so
    they were the same setting listed twice, and the value that actually wins
    (1.0, i.e. consider every feature) was absent from the grid entirely.
    n_estimators stays at the builder's 300 — more trees only ever help a
    little and cost linearly.
    """
    param_grid = {
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.5, 0.75, 1.0],
    }
    return _grid_search_best_params(
        build_rf_model(random_state=random_state), param_grid, X_train, y_train, cv_folds, "RF"
    )


def tune_xgb_hyperparams(
    X_train, y_train, cv_folds: int = DEFAULT_TUNE_CV_FOLDS, random_state: int = 0,
    n_iter: int = DEFAULT_TUNE_N_ITER,
) -> Dict:
    """Run GridSearchCV to search optimized XGBoost hyperparameters.

    54 combinations (3 sub-grids x 3 x 2 x 3), searched exhaustively. n_iter
    is accepted (but unused) for call-signature uniformity.

    learning_rate is tied to n_estimators instead of taking their product: the
    old grid swept learning_rate over 0.01-0.3 while pinning n_estimators=300,
    so its low-rate candidates were simply undertrained models the search
    would reject, burning budget. Each sub-grid below holds
    learning_rate * n_estimators roughly constant, which is what lets the
    search reach the (0.01, 1500) region that wins on this data.

    colsample_bytree is dropped, since with 9 features column subsampling has
    almost nothing to choose from, and the freed budget goes to reg_lambda.
    subsample is kept: it subsamples *rows*, so at n=460 it is a real
    regularizer rather than a feature-count question — dropping it costs about
    0.013 R^2 on Novamag. min_child_weight is left out to hold the grid near
    50 combinations; adding it back gains roughly 0.003 R^2 for 33% more fits.
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
    return _grid_search_best_params(
        build_xgb_model(random_state=random_state), param_grid, X_train, y_train, cv_folds, "XGB"
    )


# 3) Hyperparameter tuning for kernel and neural network models
def tune_svr_hyperparams(
    X_train, y_train, cv_folds: int = DEFAULT_TUNE_CV_FOLDS, random_state: int = 0,
    n_iter: int = DEFAULT_TUNE_N_ITER,
) -> Dict:
    """Run GridSearchCV to search optimized Support Vector Regressor hyperparameters.

    48 combinations, searched exhaustively. random_state and n_iter are
    accepted (but unused): SVR has a deterministic solver, and the grid is now
    enumerated rather than sampled.

    Restricted to the rbf kernel. The old grid crossed kernel with gamma, but
    gamma is meaningless for a linear kernel, so every linear candidate was
    duplicated six times and roughly half the sampled points were redundant.
    Scaled (see _scaled), rbf beats linear clearly, so linear is not worth the
    budget.
    """
    param_grid = {
        "C": [1.0, 10.0, 100.0, 1000.0],
        "gamma": ["scale", 0.01, 0.1, 1.0],
        "epsilon": [0.01, 0.05, 0.1],
    }
    return _grid_search_best_params(
        build_svr_model(), param_grid, X_train, y_train, cv_folds, "SVR", scale=True
    )


def tune_mlp_hyperparams(
    X_train, y_train, cv_folds: int = DEFAULT_TUNE_CV_FOLDS, random_state: int = 0,
    n_iter: int = DEFAULT_TUNE_N_ITER,
) -> Dict:
    """Run RandomizedSearchCV to search optimized Multi-Layer Perceptron hyperparameters."""
    param_dist = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (256, 128), (128, 64, 32)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 1e-3, 5e-3, 1e-2],
    }
    return _randomized_search_best_params(
        build_mlp_model(random_state=random_state), param_dist, X_train, y_train, cv_folds,
        random_state, "MLP", n_iter=n_iter, scale=True,
    )


# 4) Training linear models
def train_linear_regression(X_train, y_train, params: Optional[Dict] = None, random_state: int = 0):
    """Train Linear Regression (no hyperparameters, no randomness to seed)."""
    del random_state  # unused; accepted for a uniform MODEL_REGISTRY["train"] signature
    model = _scaled(build_linear_regression_model())
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train, y_train, params: Optional[Dict] = None, random_state: int = 0):
    """Train Ridge regression with searched optimized parameters or manually provided parameters."""
    del random_state  # unused; Ridge's default solver is deterministic
    if params is not None:
        ridge_model = build_ridge_model(**params)
    else:
        ridge_model = build_ridge_model(alpha=1.0)
    ridge_model = _scaled(ridge_model)
    ridge_model.fit(X_train, y_train)
    return ridge_model


def train_lasso(X_train, y_train, params: Optional[Dict] = None, random_state: int = 0):
    """Train Lasso regression with searched optimized parameters or manually provided parameters."""
    del random_state  # unused; default selection="cyclic" is deterministic
    if params is not None:
        lasso_model = build_lasso_model(**params)
    else:
        lasso_model = build_lasso_model(alpha=0.001)
    lasso_model = _scaled(lasso_model)
    lasso_model.fit(X_train, y_train)
    return lasso_model


def train_elasticnet(X_train, y_train, params: Optional[Dict] = None, random_state: int = 0):
    """Train ElasticNet regression with searched optimized parameters or manually provided parameters."""
    del random_state  # unused; default selection="cyclic" is deterministic
    if params is not None:
        enet_model = build_elasticnet_model(**params)
    else:
        enet_model = build_elasticnet_model(alpha=0.001, l1_ratio=0.1)
    enet_model = _scaled(enet_model)
    enet_model.fit(X_train, y_train)
    return enet_model


# 5) Training tree/boosting models
def train_rf(X_train, y_train, params: Optional[Dict] = None, random_state: int = 0):
    """Train a random forest with searched optimized parameters or manually provided parameters.

    Left unscaled on purpose — see _scaled.
    """
    if params is not None:
        rf_model = build_rf_model(**params, random_state=random_state)
    else:
        rf_model = build_rf_model(
            max_depth=15,
            min_samples_leaf=2,
            # 1.0, not "sqrt": with 9 features "sqrt" builds trees from 3 of
            # them and measurably underperforms (see tune_rf_hyperparams).
            max_features=1.0,
            random_state=random_state,
        )
    rf_model.fit(X_train, y_train)
    return rf_model


def train_xgb(X_train, y_train, params: Optional[Dict] = None, random_state: int = 0):
    """Train XGBoost with searched optimized parameters or manually provided parameters.

    Left unscaled on purpose — see _scaled.
    """
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
def train_svr(X_train, y_train, params: Optional[Dict] = None, random_state: int = 0):
    """Train Support Vector Regression with searched optimized parameters or manually provided parameters."""
    del random_state  # unused; SVR has no random_state (deterministic solver)
    if params is not None:
        svr_model = build_svr_model(**params)
    else:
        # rbf, not the previous linear/C=0.1 default: those were the best an
        # unscaled search could do. Scaled, rbf clearly wins (see _scaled).
        svr_model = build_svr_model(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
    svr_model = _scaled(svr_model)
    svr_model.fit(X_train, y_train)
    return svr_model


def train_mlp(X_train, y_train, params: Optional[Dict] = None, random_state: int = 0):
    """Train MLP regression with searched optimized parameters or manually provided parameters."""
    if params is not None:
        mlp_model = build_mlp_model(**params, random_state=random_state)
    else:
        mlp_model = build_mlp_model(
            hidden_layer_sizes=(256, 128),
            activation="tanh",
            alpha=1e-5,
            learning_rate_init=1e-3,
            random_state=random_state,
        )
    mlp_model = _scaled(mlp_model)
    mlp_model.fit(X_train, y_train)
    return mlp_model


MODEL_REGISTRY = build_registry([
    ModelSpec("linear", "Linear Regression", train_linear_regression),

    ModelSpec("ridge", "Ridge", train_ridge, tune_ridge_hyperparams),
    ModelSpec("lasso", "Lasso", train_lasso, tune_lasso_hyperparams),
    ModelSpec("elasticnet", "ElasticNet", train_elasticnet, tune_elasticnet_hyperparams),

    # provides_ensemble_std: RandomForestRegressor exposes `estimators_`, so the
    # UQ pipeline can read the spread of its trees. XGBoost fits one additive
    # model, not a bag of interchangeable ones, so its boosters carry no
    # comparable spread.
    ModelSpec("rf", "Random Forest", train_rf, tune_rf_hyperparams, provides_ensemble_std=True),
    ModelSpec("xgb", "XGBoost", train_xgb, tune_xgb_hyperparams),

    ModelSpec("svr", "SVR", train_svr, tune_svr_hyperparams),
    ModelSpec("mlp", "MLP", train_mlp, tune_mlp_hyperparams),
])
